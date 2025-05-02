"""
TACO-SQL Framework
=================

A comprehensive framework for Text-to-SQL generation with the following stages:
1. Question Rewrite: Clarify and normalize user queries
2. Table Linking: Identify relevant database tables
3. Query Planning: Generate execution plans
4. SQL Generation: Convert plans to SQL using various methods

SQL Generation Methods:
1. Base LLMs: Raw LLM capabilities (GPT-4, DeepSeek, etc.)
2. LLM-Based: Methods using prompt engineering (DIN-SQL, MAC-SQL)
3. SFT-Based: Fine-tuned models (CodeS-33B, Qwen2.5-Coder)
4. Hybrid: Combined approaches (CHESS, Zero-NL2SQL)
"""

# ─────────────────────────────── Imports ────────────────────────────────
from __future__ import annotations

import abc
import json
import logging
import os
import pathlib
import random
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import enum
import sqlite3
import time
import csv

import pandas as pd
import torch
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from openai import OpenAI
import yaml
import argparse
from pathlib import Path

# ─────────────────────────────── Logging ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("taco‑sql")

# ═══════════════════════ Stage 1 – Question Rewrite ══════════════════════
class QuestionRewriter(abc.ABC):
    """Interface for rewrite modules."""

    @abc.abstractmethod
    def rewrite(self, raw_query: str) -> str:
        """Return a clarified NL query."""


class EchoRewriter(QuestionRewriter):
    """Identity pass‑through (fallback)."""

    def rewrite(self, raw_query: str) -> str:  # noqa: D401
        return raw_query.strip()


class FewShotLLMRewriter(QuestionRewriter):
    """LLM‑driven rewrite with few‑shot examples & controlled decoding."""

    FEW_SHOTS: List[Tuple[str, str]] = [
        (
            "I need my employee records to finish a report. Please tell me where I can get my employee records. My employee ID is E12345, Thanks.",
            "Find storage locations for employee records with ID E12345.",
        ),
        (
            "嗨，我想知道 2023 年 5 月的销售数据，最好能按地区分组，谢谢！",
            "Retrieve 2023‑05 sales figures grouped by region.",
        ),
        (
            "Our customer table keeps crashing! What are the emails of users registered after 2024‑01‑01?",
            "List emails of customers registered after 2024‑01‑01.",
        ),
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str | None = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url or "https://api.deepseek.com")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    # ---------- prompt helpers ----------
    def _build_messages(self, query: str):
        msgs = [
            {
                "role": "system",
                "content": (
                    "You rewrite user questions for SQL retrieval. Remove irrelevant chatter, disambiguate entities, "
                    "and output one concise sentence expressing the core intent while preserving key filters."
                ),
            }
        ]
        for src, tgt in self.FEW_SHOTS:
            msgs.append({"role": "user", "content": src})
            msgs.append({"role": "assistant", "content": tgt})
        msgs.append({"role": "user", "content": query})
        return msgs

    def rewrite(self, raw_query: str) -> str:  # noqa: D401
        rsp = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(raw_query),
            temperature=self.temperature,
            top_p=self.top_p,
            seed=random.randint(1, 1000000),
        )
        rewritten = rsp.choices[0].message.content.strip()
        logger.debug("Rewriter output: %s", rewritten)
        return rewritten


# ══════════════════════ Stage 2 – Table Linking ═════════════════════════
class TableCandidate(Tuple[str, float]):
    """Helper type alias: (table_name, score)."""


class SchemaStore:
    """Lazy JSON loader mapping table_name → schema block."""

    def __init__(self, schema_json_path: str):
        self.path = schema_json_path
        with open(schema_json_path, "r", encoding="utf‑8") as f:
            self._raw = json.load(f)
        # build flat map for O(1) lookup
        self._tbl2schema: Dict[str, Dict[str, Any]] = {}
        for _, v in self._raw.items():
            if "schema" in v:
                for item in v["schema"].get("schema_items", []):
                    self._tbl2schema[item["table_name"]] = {"schema": v["schema"]}
            if "table_schema" in v and "schema" in v["table_schema"]:
                for item in v["table_schema"]["schema"].get("schema_items", []):
                    self._tbl2schema[item["table_name"]] = v["table_schema"]
        logger.info("SchemaStore loaded %d unique tables", len(self._tbl2schema))

    def get(self, table_name: str) -> Dict[str, Any] | None:
        return self._tbl2schema.get(table_name)


class TableRetriever:
    """Semantic search over table descriptions with synonym perturbation.

    Output aligns with Stage 1 (takes rewritten_query) and Stage 3a (returns a
    list of dicts: {table_name, score, schema})."""

    def __init__(
        self,
        query_model_path: str,
        table_model_path: str,
        merged_table_csv: str,
        schema_json_path: str,
    ):
        self.query_model = SentenceTransformer(query_model_path)
        self.table_model = SentenceTransformer(table_model_path)
        self.schemas = SchemaStore(schema_json_path)

        # merged_table.csv must contain columns: table_name, 表的描述, column_content_*
        self.merged_table = pd.read_csv(merged_table_csv)
        self.merged_table["table_info"] = self.merged_table.apply(self._extract_table_info, axis=1)
        table_infos = self.merged_table["table_info"].tolist()
        logger.info("Encoding %d table infos ...", len(table_infos))
        self.table_embeddings = self.table_model.encode(table_infos, convert_to_tensor=True)

    # ---- helpers ----
    @staticmethod
    def _extract_table_info(row: pd.Series) -> str:
        desc = row.get("表的描述", "")
        cols = [row[c] for c in row.index if "column_content" in c and pd.notnull(row[c])]
        return f"{desc} {' '.join(cols)}".strip()

    @staticmethod
    def _perturb_query(query: str) -> List[str]:
        words = query.split()
        perturbed = [query]
        for i, w in enumerate(words):
            syns = {l.name() for syn in wordnet.synsets(w) for l in syn.lemmas()}
            for s in syns:
                if s != w:
                    perturbed.append(" ".join(words[:i] + [s] + words[i + 1 :]))
        return perturbed

    # ---- public API ----
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple[str, float]]]]:
        """Return (top_k_candidates, debug_scores).

        *top_k_candidates* is a list of dicts each containing:
            {"table_name", "score", "schema"}
        """
        agg: Dict[str, float] = {}
        score_traces: Dict[str, List[Tuple[str, float]]] = {}

        for pq in self._perturb_query(query):
            q_emb = self.query_model.encode(pq, convert_to_tensor=True)
            hits = util.semantic_search(q_emb, self.table_embeddings, top_k=k)[0]
            for hit in hits:
                idx = hit["corpus_id"] if isinstance(hit["corpus_id"], int) else hit["corpus_id"]["idx"]
                tname = self.merged_table.iloc[idx]["table_name"]
                score = float(hit["score"])
                agg[tname] = agg.get(tname, 0.0) + score
                score_traces.setdefault(tname, []).append((pq, score))

        ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:k]
        top_candidates: List[Dict[str, Any]] = []
        for tname, score in ranked:
            schema = self.schemas.get(tname)
            top_candidates.append({"table_name": tname, "score": score, "schema": schema})
        logger.debug("Top tables: %s", [(c["table_name"], round(c["score"], 4)) for c in top_candidates])
        return top_candidates, score_traces


# ═════════════ Stage 3a – Query Planner (baseline‑agnostic) ═════════════
@dataclass
class ExecStep:
    """A structured sub‑query unit used by downstream generators."""

    step_id: int
    operation: str  # e.g., SELECT, JOIN, FILTER, GROUPBY
    table: str | None = None
    expression: str | None = None  # textual description of the sub‑query

    def to_dict(self):
        return {
            "id": self.step_id,
            "op": self.operation,
            "table": self.table,
            "expr": self.expression,
        }


class HeuristicPlanner:
    """Lightweight rule‑based planner – no external calls.

    * Splits conditions by common keywords (where, after, before, greater than, etc.).
    * Detects obvious join hints if multiple tables are selected.
    """

    _COND_PTRN = re.compile(r"(after|before|>=|<=|greater than|less than|equals|=|>|<|between)", re.I)

    def plan(self, rewritten_query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        tables = [c["table_name"] for c in candidates]
        steps: List[ExecStep] = []

        # step 0: initial intent
        steps.append(ExecStep(0, "INTENT", expression=rewritten_query))

        # step 1‑n: per‑table selection skeleton
        for i, t in enumerate(tables, start=1):
            steps.append(ExecStep(i, "SELECT", table=t, expression=f"Select * from {t}"))

        # naive condition extraction
        conds = self._COND_PTRN.split(rewritten_query)
        if len(conds) > 1:
            steps.append(
                ExecStep(len(steps), "FILTER", expression=" ".join(conds[-2:]).strip())
            )

        # simple join if >1 table
        if len(tables) > 1:
            steps.append(
                ExecStep(len(steps), "JOIN", expression="Join tables on common keys (heuristic)"),
            )

        plan_dict = {
            "tables": tables,
            "steps": [s.to_dict() for s in steps],
            "schemas": {c["table_name"]: c["schema"] for c in candidates if c["schema"]},
        }
        return plan_dict


class LLMJSONPlanner(HeuristicPlanner):
    """Few‑shot prompt to LLM that returns a JSON execution plan.

    Falls back to `HeuristicPlanner` if the LLM fails to produce valid JSON.
    """

    FEW_SHOT_CONTEXT = textwrap.dedent(
        """
        You are an expert data engineer. Given a natural‑language query and a list of candidate
        tables (with brief descriptions), break the query into an ordered JSON execution plan.
        Each step should have fields: id, op (SELECT/FILTER/JOIN/GROUPBY/etc.), table (nullable), expr.
        Example:
        NL: "List employee names hired after 2020‑01‑01 along with their department names."
        CANDIDATE_TABLES: [employees, departments]
        OUTPUT_JSON:
        {"tables": ["employees", "departments"],
         "steps": [
            {"id":0, "op":"INTENT", "expr":"list employee names with department"},
            {"id":1, "op":"SELECT", "table":"employees", "expr":"id, name, dept_id"},
            {"id":2, "op":"FILTER", "expr":"hire_date > '2020-01-01'"},
            {"id":3, "op":"SELECT", "table":"departments", "expr":"id, dept_name"},
            {"id":4, "op":"JOIN", "expr":"employees.dept_id = departments.id"}
         ]}
        """
    )

    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url or "https://api.deepseek.com")
        self.model = model

    def plan(self, rewritten_query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        cand_names = [c["table_name"] for c in candidates]
        prompt = (
            self.FEW_SHOT_CONTEXT + f"NL: {rewritten_query} CANDIDATE_TABLES: {cand_names} OUTPUT_JSON:"
        )
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            txt = rsp.choices[0].message.content.strip()
            plan = json.loads(txt)
            plan["schemas"] = {c["table_name"]: c["schema"] for c in candidates if c["schema"]}
            return plan
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM planner failed (%s) – falling back to heuristic.", e)
            return super().plan(rewritten_query, candidates)

# ══════════════════════ Stage 4 – SQL Generation ════════════════════════

# 4.1 Base Classes and Types
class SQLGenerationType(enum.Enum):
    """SQL Generation method types."""
    # Base LLM Methods
    BASE_LLM = "base_llm"           # Raw LLM capabilities
    
    # LLM-Based Methods
    DIN_SQL = "din_sql"             # DIN-SQL method
    MAC_SQL = "mac_sql"             # MAC-SQL method
    
    # SFT-Based Methods
    SFT_BASED = "sft_based"         # Base SFT method
    CODES_33B = "codes_33b"         # CodeS-33B model
    QWEN_CODER = "qwen_coder"       # Qwen2.5-Coder model
    
    # Hybrid Methods
    CHESS = "chess"                 # CHESS method
    ZERO_NL2SQL = "zero_nl2sql"     # Zero-NL2SQL method
    HYBRID = "hybrid"               # Combined approaches

class BaseSQLGenerator(abc.ABC):
    """Base class for all SQL generators."""
    
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def generate(self, plan: Dict[str, Any]) -> str:
        """Generate SQL from a query plan."""
        pass

class SQLGeneratorConfig:
    """Configuration for SQL generators."""
    def __init__(
        self,
        generation_type: SQLGenerationType,
        model_name: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        model_path: str | None = None,
        device: str = "cuda:0",
        temperature: float = 0.1,
        top_p: float = 0.9,
        # TACO-Benchmark specific configs
        data_path: str | None = None,
        tables_json_path: str | None = None,
        dataset_name: str = "us"  # 默认使用 US 数据集
    ):
        self.generation_type = generation_type
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        # TACO-Benchmark specific configs
        self.data_path = data_path
        self.tables_json_path = tables_json_path
        
        # 验证数据集名称
        valid_datasets = ["us", "beijing", "smartcity"]
        if dataset_name not in valid_datasets:
            raise ValueError(f"Invalid dataset name: {dataset_name}. Must be one of: {valid_datasets}")
        self.dataset_name = dataset_name

# 4.2 Base LLM Methods
class BaseLLMGenerator(BaseSQLGenerator):
    """Base class for LLM-based SQL generators."""
    
    def __init__(self, config: SQLGeneratorConfig):
        super().__init__(f"BaseLLM-{config.model_name}")
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model_name
        self.temperature = config.temperature
        self.top_p = config.top_p

    def _build_prompt(self, plan: Dict[str, Any]) -> str:
        return f"""You are an expert SQL generator. Given a query plan, generate a SQLite-compatible SQL statement.
Plan: {json.dumps(plan, ensure_ascii=False)}

Rules:
- Use correct tables and columns from the schema
- Use double quotes for identifiers with spaces/special characters
- Return only the SQL statement"""

    def generate(self, plan: Dict[str, Any]) -> str:
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful SQL assistant."},
                    {"role": "user", "content": self._build_prompt(plan)}
                ],
                temperature=self.temperature,
                top_p=self.top_p
            )
            sql = rsp.choices[0].message.content.strip()
            return sql
        except Exception as e:
            logger.error(f"SQL generation failed ({e})")
            return f"SELECT * FROM {plan['tables'][0]}"

class DeepSeekGenerator(BaseLLMGenerator):
    """DeepSeek-specific SQL generator."""
    def __init__(self, config: SQLGeneratorConfig):
        super().__init__(config)
        self.name = "DeepSeek"

# 4.3 LLM-Based Methods
class DINSQLGenerator(BaseSQLGenerator):
    """DIN-SQL style generator with question classification and multi-step generation."""
    
    def __init__(self, config: SQLGeneratorConfig):
        super().__init__("DIN-SQL")
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model_name
        self.temperature = config.temperature
        self.classifier = QuestionClassifier(config.api_key, config.model_name, config.base_url)
        
        # 初始化SQL验证器
        self.validator = SQLValidator()
        
        # 初始化SQL修复器
        self.fixer = SQLFixer(
            api_key=config.api_key,
            model_name=config.model_name,
            base_url=config.base_url
        )
        
    def _build_prompt(self, plan: Dict[str, Any], category: str) -> str:
        """构建提示模板"""
        base_prompt = f"""You are an expert SQL generator. Given a query plan, generate a SQLite-compatible SQL statement.
Plan: {json.dumps(plan, ensure_ascii=False)}

Rules:
- Use correct tables and columns from the schema
- Use double quotes for identifiers with spaces/special characters
- Return only the SQL statement
- Follow SQLite syntax rules
- Handle NULL values appropriately
- Use proper JOIN syntax
- Include necessary WHERE clauses
- Use appropriate aggregation functions
- Format dates correctly
"""
        
        if category == "EASY":
            return base_prompt + """
This is a simple query - focus on:
1. Basic SELECT and WHERE clauses
2. Simple column selection
3. Basic filtering conditions
4. No complex joins or subqueries needed
"""
        elif category == "NON-NESTED":
            return base_prompt + """
This query may need:
1. Multiple table JOINs
2. GROUP BY clauses
3. Complex WHERE conditions
4. Basic aggregations
5. ORDER BY clauses
"""
        else:  # NESTED
            return base_prompt + """
This query may need:
1. Subqueries or CTEs
2. Complex aggregations
3. Window functions
4. Multiple levels of nesting
5. Complex JOIN conditions
"""
            
    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        """验证SQL语法和结构"""
        # 基本语法检查
        if not sql.strip().upper().startswith("SELECT"):
            return False, "SQL must start with SELECT"
            
        # 检查SQL语法
        try:
            sql = sql.strip()
            if not sql.endswith(";"):
                sql += ";"
                
            # 使用SQLValidator进行验证
            is_valid, error = self.validator.validate(sql)
            if not is_valid:
                return False, error
                
            return True, ""
            
        except Exception as e:
            return False, str(e)
        
    def _fix_sql(self, sql: str, error: str) -> str:
        """修复SQL错误"""
        try:
            # 使用SQLFixer修复SQL
            fixed_sql = self.fixer.fix(sql, error)
            
            # 验证修复后的SQL
            is_valid, new_error = self._validate_sql(fixed_sql)
            if is_valid:
                return fixed_sql
            else:
                # 如果修复后仍然无效,尝试再次修复
                return self.fixer.fix(fixed_sql, new_error)
                
        except Exception as e:
            logger.warning(f"SQL fix failed ({e})")
            return sql
            
    def _optimize_sql(self, sql: str) -> str:
        """优化SQL查询"""
        try:
            # 移除不必要的空格
            sql = re.sub(r'\s+', ' ', sql).strip()
            
            # 标准化引号
            sql = sql.replace('"', '"').replace('"', '"')
            
            # 标准化关键字大小写
            keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER']
            for keyword in keywords:
                sql = re.sub(rf'\b{keyword}\b', keyword, sql, flags=re.IGNORECASE)
                
            return sql
            
        except Exception as e:
            logger.warning(f"SQL optimization failed ({e})")
            return sql
            
    def generate(self, plan: Dict[str, Any]) -> str:
        """使用DIN-SQL方法生成SQL"""
        try:
            # 1. 问题分类
            query = plan.get("steps", [{}])[0].get("expr", "")
            category = self.classifier.classify(query)
            logger.info(f"Question category: {category}")
            
            # 2. 构建提示
            prompt = self._build_prompt(plan, category)
            
            # 3. 生成初始SQL
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            sql = rsp.choices[0].message.content.strip()
            
            # 4. 验证SQL
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.warning(f"Initial SQL validation failed: {error}")
                sql = self._fix_sql(sql, error)
                
            # 5. 优化SQL
            sql = self._optimize_sql(sql)
            
            # 6. 最终验证
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.error(f"Final SQL validation failed: {error}")
                # 如果最终验证失败,返回一个简单的查询
                return f"SELECT * FROM {plan['tables'][0]}"
                
            return sql
            
        except Exception as e:
            logger.error(f"SQL generation failed ({e})")
            return f"SELECT * FROM {plan['tables'][0]}"

class SQLValidator:
    """SQL验证器"""
    
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        
    def validate(self, sql: str) -> Tuple[bool, str]:
        """验证SQL语法"""
        try:
            # 尝试解析SQL
            self.conn.execute("EXPLAIN " + sql)
            return True, ""
        except sqlite3.Error as e:
            return False, str(e)
            
    def __del__(self):
        self.conn.close()

class SQLFixer:
    """SQL修复器"""
    
    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        
    def fix(self, sql: str, error: str) -> str:
        """修复SQL错误"""
        prompt = f"""The following SQL has an error: {error}
SQL: {sql}

Please fix the SQL and return only the corrected version. Follow these rules:
1. Keep the same query intent
2. Fix syntax errors
3. Use correct table and column names
4. Follow SQLite syntax
5. Return only the SQL statement"""

        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"SQL fix failed ({e})")
            return sql

class MACSQLGenerator(BaseSQLGenerator):
    """MAC-SQL style generator with multi-agent collaboration."""
    
    def __init__(self, config: SQLGeneratorConfig):
        super().__init__("MAC-SQL")
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model_name
        self.temperature = config.temperature
        
        # 初始化组件
        self.selector = Selector(config.api_key, config.model_name, config.base_url)
        self.decomposer = Decomposer(config.api_key, config.model_name, config.base_url)
        self.refiner = Refiner(config.api_key, config.model_name, config.base_url)
        
        # 初始化验证器
        self.validator = SQLValidator()
        
        # 加载数据库信息
        self.data_path = config.data_path
        self.tables_json_path = config.tables_json_path
        self.dataset_name = config.dataset_name
        self._load_database_info()
        
    def _load_database_info(self):
        """加载数据库信息"""
        try:
            # 加载表信息
            with open(self.tables_json_path, 'r', encoding='utf-8') as f:
                self.tables_info = json.load(f)
                
            # 加载数据库连接
            self.conn = sqlite3.connect(self.data_path)
            self.cursor = self.conn.cursor()
            
            # 获取表结构
            self.schema = {}
            for table in self.tables_info:
                self.cursor.execute(f"PRAGMA table_info({table['name']})")
                columns = self.cursor.fetchall()
                self.schema[table['name']] = [
                    {
                        'name': col[1],
                        'type': col[2],
                        'nullable': not col[3],
                        'default': col[4],
                        'pk': col[5]
                    }
                    for col in columns
                ]
                
        except Exception as e:
            logger.error(f"Failed to load database info: {e}")
            raise
            
    def _build_selector_prompt(self, query: str) -> str:
        """构建选择器提示"""
        return f"""Given the following question and database schema, select the most relevant tables and columns.

Question: {query}

Database Schema:
{json.dumps(self.schema, indent=2, ensure_ascii=False)}

Rules:
1. Select only tables and columns that are directly relevant to answering the question
2. Consider both explicit and implicit relationships
3. Include primary and foreign keys for joins
4. Consider data types and constraints
5. Return a JSON object with:
   - tables: list of selected table names
   - columns: dict mapping table names to lists of column names
   - relationships: list of table relationships (if any)

Return only the JSON object."""

    def _build_decomposer_prompt(self, query: str, selected_info: Dict[str, Any]) -> str:
        """构建分解器提示"""
        return f"""Given the following question and selected database information, decompose it into simpler sub-queries.

Question: {query}

Selected Information:
{json.dumps(selected_info, indent=2, ensure_ascii=False)}

Rules:
1. Break down complex queries into simpler steps
2. Each step should be self-contained and meaningful
3. Consider dependencies between steps
4. Include necessary joins and conditions
5. Return a JSON object with:
   - steps: list of sub-queries with their descriptions
   - dependencies: list of step dependencies
   - final_query: description of how to combine results

Return only the JSON object."""

    def _build_refiner_prompt(self, query: str, decomposition: Dict[str, Any]) -> str:
        """构建优化器提示"""
        return f"""Given the following question and query decomposition, generate the final SQL query.

Question: {query}

Decomposition:
{json.dumps(decomposition, indent=2, ensure_ascii=False)}

Rules:
1. Generate SQLite-compatible SQL
2. Use proper table and column names
3. Include all necessary joins
4. Add appropriate conditions
5. Handle NULL values
6. Use correct aggregation functions
7. Format dates properly
8. Return only the SQL statement"""

    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        """验证SQL语法和结构"""
        try:
            # 基本语法检查
            if not sql.strip().upper().startswith("SELECT"):
                return False, "SQL must start with SELECT"
                
            # 使用SQLValidator验证
            is_valid, error = self.validator.validate(sql)
            if not is_valid:
                return False, error
                
            # 检查表名和列名
            tables = re.findall(r'FROM\s+([^\s,;]+)|JOIN\s+([^\s,;]+)', sql, re.IGNORECASE)
            tables = [t[0] or t[1] for t in tables]
            
            for table in tables:
                if table not in self.schema:
                    return False, f"Table {table} not found in schema"
                    
            return True, ""
            
        except Exception as e:
            return False, str(e)
            
    def _optimize_sql(self, sql: str) -> str:
        """优化SQL查询"""
        try:
            # 移除不必要的空格
            sql = re.sub(r'\s+', ' ', sql).strip()
            
            # 标准化引号
            sql = sql.replace('"', '"').replace('"', '"')
            
            # 标准化关键字大小写
            keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER']
            for keyword in keywords:
                sql = re.sub(rf'\b{keyword}\b', keyword, sql, flags=re.IGNORECASE)
                
            # 优化JOIN顺序
            # TODO: 实现JOIN顺序优化
            
            return sql
            
        except Exception as e:
            logger.warning(f"SQL optimization failed ({e})")
            return sql
            
    def generate(self, plan: Dict[str, Any]) -> str:
        """使用MAC-SQL方法生成SQL"""
        try:
            query = plan.get("steps", [{}])[0].get("expr", "")
            
            # 1. 选择相关表和列
            selected_info = self.selector.select(query, self._build_selector_prompt(query))
            logger.info(f"Selected tables and columns: {selected_info}")
            
            # 2. 分解查询
            decomposition = self.decomposer.decompose(query, selected_info, self._build_decomposer_prompt(query, selected_info))
            logger.info(f"Query decomposition: {decomposition}")
            
            # 3. 生成SQL
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self._build_refiner_prompt(query, decomposition)}],
                temperature=self.temperature
            )
            sql = rsp.choices[0].message.content.strip()
            
            # 4. 验证SQL
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.warning(f"SQL validation failed: {error}")
                # 尝试修复SQL
                sql = self._fix_sql(sql, error)
                
            # 5. 优化SQL
            sql = self._optimize_sql(sql)
            
            # 6. 最终验证
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.error(f"Final SQL validation failed: {error}")
                # 如果最终验证失败,返回一个简单的查询
                return f"SELECT * FROM {selected_info['tables'][0]}"
                
            return sql
            
        except Exception as e:
            logger.error(f"SQL generation failed ({e})")
            return f"SELECT * FROM {plan['tables'][0]}"
            
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            logger.warning(f"Failed to close database connection: {e}")

class Selector:
    """表选择器组件"""
    
    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        
    def select(self, query: str, prompt: str) -> Dict[str, Any]:
        """选择相关表和列"""
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return json.loads(rsp.choices[0].message.content)
        except Exception as e:
            logger.error(f"Table selection failed ({e})")
            return {"tables": [], "columns": {}, "relationships": []}

class Decomposer:
    """查询分解器组件"""
    
    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        
    def decompose(self, query: str, selected_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """分解复杂查询"""
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return json.loads(rsp.choices[0].message.content)
        except Exception as e:
            logger.error(f"Query decomposition failed ({e})")
            return {"steps": [], "dependencies": [], "final_query": ""}

class Refiner:
    """SQL优化器组件"""
    
    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        
    def refine(self, query: str, decomposition: Dict[str, Any], prompt: str) -> str:
        """优化SQL查询"""
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"SQL refinement failed ({e})")
            return f"SELECT * FROM {decomposition['steps'][0]['table']}"

# 4.4 SFT-Based Methods
class SFTBasedGenerator(BaseSQLGenerator):
    """Base class for SFT-based SQL generators."""
    
    def __init__(self, config: SQLGeneratorConfig):
        super().__init__(f"SFT-{pathlib.Path(config.model_path).stem}")
        import transformers
        self.pipe = transformers.pipeline(
            "text-generation",
            model=config.model_path,
            device=config.device,
            model_kwargs={"torch_dtype": torch.float16}
        )

    def generate(self, plan: Dict[str, Any]) -> str:
        prompt = f"Translate the following query plan into a SQLite SQL statement: {json.dumps(plan, ensure_ascii=False)}"
        try:
            out = self.pipe(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
            sql = out[len(prompt):].strip().split("\n")[0]
            return sql
        except Exception as e:
            logger.error(f"SQL generation failed ({e})")
            return f"SELECT * FROM {plan['tables'][0]}"

class CodeS33BGenerator(SFTBasedGenerator):
    """CodeS-33B SFT model generator."""
    def __init__(self, config: SQLGeneratorConfig):
        super().__init__(config)
        self.name = "CodeS-33B"

class QwenCoderGenerator(SFTBasedGenerator):
    """Qwen2.5-Coder-32B SFT model generator."""
    def __init__(self, config: SQLGeneratorConfig):
        super().__init__(config)
        self.name = "Qwen-Coder-32B"

# 4.5 Hybrid Methods
class CHESSGenerator(BaseSQLGenerator):
    """CHESS style generator with query plan based generation."""
    
    def __init__(self, config: SQLGeneratorConfig):
        super().__init__("CHESS")
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model_name
        self.temperature = config.temperature
        
        # 初始化验证器
        self.validator = SQLValidator()
        
        # 加载数据库信息
        self.data_path = config.data_path
        self.tables_json_path = config.tables_json_path
        self.dataset_name = config.dataset_name
        self._load_database_info()
        
    def _load_database_info(self):
        """加载数据库信息"""
        try:
            # 加载表信息
            with open(self.tables_json_path, 'r', encoding='utf-8') as f:
                self.tables_info = json.load(f)
                
            # 加载数据库连接
            self.conn = sqlite3.connect(self.data_path)
            self.cursor = self.conn.cursor()
            
            # 获取表结构
            self.schema = {}
            for table in self.tables_info:
                self.cursor.execute(f"PRAGMA table_info({table['name']})")
                columns = self.cursor.fetchall()
                self.schema[table['name']] = [
                    {
                        'name': col[1],
                        'type': col[2],
                        'nullable': not col[3],
                        'default': col[4],
                        'pk': col[5]
                    }
                    for col in columns
                ]
                
        except Exception as e:
            logger.error(f"Failed to load database info: {e}")
            raise
            
    def _build_query_plan_prompt(self, query: str) -> str:
        """构建查询计划提示"""
        return f"""You are a database administrator. Given a question, generate a detailed query plan.

Question: {query}

Database Schema:
{json.dumps(self.schema, indent=2, ensure_ascii=False)}

Rules for Database Administrators:
1. Always verify table and column existence
2. Consider data types and constraints
3. Use appropriate indexes
4. Optimize JOIN operations
5. Handle NULL values properly
6. Consider query performance
7. Use proper date/time functions
8. Handle string operations carefully
9. Consider data volume
10. Use appropriate aggregation functions
11. Consider query complexity

Steps to Generate Query Plan:
1. Analyze the question requirements
2. Identify required tables and columns
3. Determine necessary JOINs
4. Specify filtering conditions
5. Define aggregation needs
6. Consider sorting requirements
7. Plan subqueries if needed
8. Optimize the execution plan

Input Information:
- Database Schema: {json.dumps(self.schema, indent=2, ensure_ascii=False)}
- Question: {query}

Please provide a detailed query plan following the above rules and steps."""

    def _extract_sql_from_response(self, response: str) -> str:
        """从响应中提取SQL语句"""
        try:
            # 尝试找到SQL语句
            sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
                
            # 如果没有找到SQL代码块,尝试直接提取
            lines = response.split('\n')
            sql_lines = []
            in_sql = False
            
            for line in lines:
                if line.strip().upper().startswith('SELECT'):
                    in_sql = True
                if in_sql:
                    sql_lines.append(line)
                if in_sql and line.strip().endswith(';'):
                    break
                    
            if sql_lines:
                return '\n'.join(sql_lines).strip()
                
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Failed to extract SQL from response: {e}")
            return response.strip()
            
    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        """验证SQL语法和结构"""
        try:
            # 基本语法检查
            if not sql.strip().upper().startswith("SELECT"):
                return False, "SQL must start with SELECT"
                
            # 使用SQLValidator验证
            is_valid, error = self.validator.validate(sql)
            if not is_valid:
                return False, error
                
            # 检查表名和列名
            tables = re.findall(r'FROM\s+([^\s,;]+)|JOIN\s+([^\s,;]+)', sql, re.IGNORECASE)
            tables = [t[0] or t[1] for t in tables]
            
            for table in tables:
                if table not in self.schema:
                    return False, f"Table {table} not found in schema"
                    
            return True, ""
            
        except Exception as e:
            return False, str(e)
            
    def _optimize_sql(self, sql: str) -> str:
        """优化SQL查询"""
        try:
            # 移除不必要的空格
            sql = re.sub(r'\s+', ' ', sql).strip()
            
            # 标准化引号
            sql = sql.replace('"', '"').replace('"', '"')
            
            # 标准化关键字大小写
            keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER']
            for keyword in keywords:
                sql = re.sub(rf'\b{keyword}\b', keyword, sql, flags=re.IGNORECASE)
                
            # 优化JOIN顺序
            # TODO: 实现JOIN顺序优化
            
            return sql
            
        except Exception as e:
            logger.warning(f"SQL optimization failed ({e})")
            return sql
            
    def generate(self, plan: Dict[str, Any]) -> str:
        """使用CHESS方法生成SQL"""
        try:
            query = plan.get("steps", [{}])[0].get("expr", "")
            
            # 1. 生成查询计划
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self._build_query_plan_prompt(query)}],
                temperature=self.temperature
            )
            query_plan = rsp.choices[0].message.content
            
            # 2. 从查询计划中提取SQL
            sql = self._extract_sql_from_response(query_plan)
            
            # 3. 验证SQL
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.warning(f"SQL validation failed: {error}")
                # 尝试修复SQL
                sql = self._fix_sql(sql, error)
                
            # 4. 优化SQL
            sql = self._optimize_sql(sql)
            
            # 5. 最终验证
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.error(f"Final SQL validation failed: {error}")
                # 如果最终验证失败,返回一个简单的查询
                return f"SELECT * FROM {plan['tables'][0]}"
                
            return sql
            
        except Exception as e:
            logger.error(f"SQL generation failed ({e})")
            return f"SELECT * FROM {plan['tables'][0]}"
            
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            logger.warning(f"Failed to close database connection: {e}")

class ZeroNL2SQLGenerator(BaseSQLGenerator):
    """Zero-NL2SQL style generator with small model structure generation and LLM refinement."""
    
    def __init__(self, config: SQLGeneratorConfig):
        super().__init__("Zero-NL2SQL")
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model_name
        self.temperature = config.temperature
        
        # 初始化验证器
        self.validator = SQLValidator()
        
        # 加载数据库信息
        self.data_path = config.data_path
        self.tables_json_path = config.tables_json_path
        self.dataset_name = config.dataset_name
        self._load_database_info()
        
    def _load_database_info(self):
        """加载数据库信息"""
        try:
            # 加载表信息
            with open(self.tables_json_path, 'r', encoding='utf-8') as f:
                self.tables_info = json.load(f)
                
            # 加载数据库连接
            self.conn = sqlite3.connect(self.data_path)
            self.cursor = self.conn.cursor()
            
            # 获取表结构
            self.schema = {}
            for table in self.tables_info:
                self.cursor.execute(f"PRAGMA table_info({table['name']})")
                columns = self.cursor.fetchall()
                self.schema[table['name']] = [
                    {
                        'name': col[1],
                        'type': col[2],
                        'nullable': not col[3],
                        'default': col[4],
                        'pk': col[5]
                    }
                    for col in columns
                ]
                
        except Exception as e:
            logger.error(f"Failed to load database info: {e}")
            raise
            
    def _build_structure_prompt(self, query: str) -> str:
        """构建结构生成提示"""
        return f"""Given the following question and database schema, generate a SQL query structure.

Question: {query}

Database Schema:
{json.dumps(self.schema, indent=2, ensure_ascii=False)}

Rules:
1. Focus on query structure, not specific values
2. Include table names and column names
3. Specify JOIN types and conditions
4. Define WHERE clauses
5. List required aggregations
6. Note any subqueries needed
7. Consider sorting requirements
8. Return a JSON object with:
   - tables: list of tables to use
   - columns: dict mapping tables to columns
   - joins: list of join conditions
   - where: list of where conditions
   - group_by: list of grouping columns
   - having: list of having conditions
   - order_by: list of ordering columns
   - subqueries: list of subquery structures

Return only the JSON object."""

    def _build_refinement_prompt(self, query: str, structure: Dict[str, Any]) -> str:
        """构建优化提示"""
        return f"""Given the following question and SQL structure, generate the final SQL query.

Question: {query}

SQL Structure:
{json.dumps(structure, indent=2, ensure_ascii=False)}

Rules:
1. Generate SQLite-compatible SQL
2. Use proper table and column names
3. Include all necessary joins
4. Add appropriate conditions
5. Handle NULL values
6. Use correct aggregation functions
7. Format dates properly
8. Return only the SQL statement"""

    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        """验证SQL语法和结构"""
        try:
            # 基本语法检查
            if not sql.strip().upper().startswith("SELECT"):
                return False, "SQL must start with SELECT"
                
            # 使用SQLValidator验证
            is_valid, error = self.validator.validate(sql)
            if not is_valid:
                return False, error
                
            # 检查表名和列名
            tables = re.findall(r'FROM\s+([^\s,;]+)|JOIN\s+([^\s,;]+)', sql, re.IGNORECASE)
            tables = [t[0] or t[1] for t in tables]
            
            for table in tables:
                if table not in self.schema:
                    return False, f"Table {table} not found in schema"
                    
            return True, ""
            
        except Exception as e:
            return False, str(e)
            
    def _optimize_sql(self, sql: str) -> str:
        """优化SQL查询"""
        try:
            # 移除不必要的空格
            sql = re.sub(r'\s+', ' ', sql).strip()
            
            # 标准化引号
            sql = sql.replace('"', '"').replace('"', '"')
            
            # 标准化关键字大小写
            keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER']
            for keyword in keywords:
                sql = re.sub(rf'\b{keyword}\b', keyword, sql, flags=re.IGNORECASE)
                
            # 优化JOIN顺序
            # TODO: 实现JOIN顺序优化
            
            return sql
            
        except Exception as e:
            logger.warning(f"SQL optimization failed ({e})")
            return sql
            
    def generate(self, plan: Dict[str, Any]) -> str:
        """使用Zero-NL2SQL方法生成SQL"""
        try:
            query = plan.get("steps", [{}])[0].get("expr", "")
            
            # 1. 生成查询结构
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self._build_structure_prompt(query)}],
                temperature=0.0
            )
            structure = json.loads(rsp.choices[0].message.content)
            logger.info(f"Generated query structure: {structure}")
            
            # 2. 优化查询结构
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self._build_refinement_prompt(query, structure)}],
                temperature=self.temperature
            )
            sql = rsp.choices[0].message.content.strip()
            
            # 3. 验证SQL
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.warning(f"SQL validation failed: {error}")
                # 尝试修复SQL
                sql = self._fix_sql(sql, error)
                
            # 4. 优化SQL
            sql = self._optimize_sql(sql)
            
            # 5. 最终验证
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.error(f"Final SQL validation failed: {error}")
                # 如果最终验证失败,返回一个简单的查询
                return f"SELECT * FROM {plan['tables'][0]}"
                
            return sql
            
        except Exception as e:
            logger.error(f"SQL generation failed ({e})")
            return f"SELECT * FROM {plan['tables'][0]}"
            
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            logger.warning(f"Failed to close database connection: {e}")

class HybridGenerator(BaseSQLGenerator):
    """Hybrid style generator combining multiple approaches."""
    
    def __init__(self, config: SQLGeneratorConfig):
        super().__init__("Hybrid")
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model_name
        self.temperature = config.temperature
        
        # 初始化验证器
        self.validator = SQLValidator()
        
        # 加载数据库信息
        self.data_path = config.data_path
        self.tables_json_path = config.tables_json_path
        self.dataset_name = config.dataset_name
        self._load_database_info()
        
        # 初始化其他生成器
        self.din_generator = DINSQLGenerator(config)
        self.mac_generator = MACSQLGenerator(config)
        self.chess_generator = CHESSGenerator(config)
        self.zero_generator = ZeroNL2SQLGenerator(config)
        
    def _load_database_info(self):
        """加载数据库信息"""
        try:
            # 加载表信息
            with open(self.tables_json_path, 'r', encoding='utf-8') as f:
                self.tables_info = json.load(f)
                
            # 加载数据库连接
            self.conn = sqlite3.connect(self.data_path)
            self.cursor = self.conn.cursor()
            
            # 获取表结构
            self.schema = {}
            for table in self.tables_info:
                self.cursor.execute(f"PRAGMA table_info({table['name']})")
                columns = self.cursor.fetchall()
                self.schema[table['name']] = [
                    {
                        'name': col[1],
                        'type': col[2],
                        'nullable': not col[3],
                        'default': col[4],
                        'pk': col[5]
                    }
                    for col in columns
                ]
                
        except Exception as e:
            logger.error(f"Failed to load database info: {e}")
            raise
            
    def _build_ensemble_prompt(self, query: str, sql_list: List[str]) -> str:
        """构建集成提示"""
        return f"""Given the following question and multiple SQL queries, generate the best SQL query.

Question: {query}

Generated SQL Queries:
{json.dumps(sql_list, indent=2, ensure_ascii=False)}

Rules:
1. Analyze each SQL query
2. Identify the strengths and weaknesses
3. Combine the best parts
4. Fix any issues
5. Ensure SQLite compatibility
6. Handle NULL values
7. Use proper joins
8. Return only the final SQL statement"""

    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        """验证SQL语法和结构"""
        try:
            # 基本语法检查
            if not sql.strip().upper().startswith("SELECT"):
                return False, "SQL must start with SELECT"
                
            # 使用SQLValidator验证
            is_valid, error = self.validator.validate(sql)
            if not is_valid:
                return False, error
                
            # 检查表名和列名
            tables = re.findall(r'FROM\s+([^\s,;]+)|JOIN\s+([^\s,;]+)', sql, re.IGNORECASE)
            tables = [t[0] or t[1] for t in tables]
            
            for table in tables:
                if table not in self.schema:
                    return False, f"Table {table} not found in schema"
                    
            return True, ""
            
        except Exception as e:
            return False, str(e)
            
    def _optimize_sql(self, sql: str) -> str:
        """优化SQL查询"""
        try:
            # 移除不必要的空格
            sql = re.sub(r'\s+', ' ', sql).strip()
            
            # 标准化引号
            sql = sql.replace('"', '"').replace('"', '"')
            
            # 标准化关键字大小写
            keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER']
            for keyword in keywords:
                sql = re.sub(rf'\b{keyword}\b', keyword, sql, flags=re.IGNORECASE)
                
            # 优化JOIN顺序
            # TODO: 实现JOIN顺序优化
            
            return sql
            
        except Exception as e:
            logger.warning(f"SQL optimization failed ({e})")
            return sql
            
    def _compare_sql(self, sql1: str, sql2: str) -> bool:
        """比较两个SQL查询的相似度"""
        try:
            # 标准化SQL
            sql1 = self._optimize_sql(sql1)
            sql2 = self._optimize_sql(sql2)
            
            # 提取关键字和操作符
            keywords1 = set(re.findall(r'\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY|HAVING|JOIN|LEFT|RIGHT|INNER|OUTER)\b', sql1, re.IGNORECASE))
            keywords2 = set(re.findall(r'\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY|HAVING|JOIN|LEFT|RIGHT|INNER|OUTER)\b', sql2, re.IGNORECASE))
            
            # 提取表名
            tables1 = set(re.findall(r'FROM\s+([^\s,;]+)|JOIN\s+([^\s,;]+)', sql1, re.IGNORECASE))
            tables2 = set(re.findall(r'FROM\s+([^\s,;]+)|JOIN\s+([^\s,;]+)', sql2, re.IGNORECASE))
            
            # 计算相似度
            keyword_similarity = len(keywords1 & keywords2) / len(keywords1 | keywords2) if keywords1 | keywords2 else 0
            table_similarity = len(tables1 & tables2) / len(tables1 | tables2) if tables1 | tables2 else 0
            
            return (keyword_similarity + table_similarity) / 2 > 0.5
            
        except Exception as e:
            logger.warning(f"SQL comparison failed ({e})")
            return False
            
    def generate(self, plan: Dict[str, Any]) -> str:
        """使用混合方法生成SQL"""
        try:
            query = plan.get("steps", [{}])[0].get("expr", "")
            
            # 1. 使用不同的生成器生成SQL
            sql_list = []
            
            # DIN-SQL
            try:
                din_sql = self.din_generator.generate(plan)
                if din_sql:
                    sql_list.append(din_sql)
            except Exception as e:
                logger.warning(f"DIN-SQL generation failed: {e}")
                
            # MAC-SQL
            try:
                mac_sql = self.mac_generator.generate(plan)
                if mac_sql:
                    sql_list.append(mac_sql)
            except Exception as e:
                logger.warning(f"MAC-SQL generation failed: {e}")
                
            # CHESS
            try:
                chess_sql = self.chess_generator.generate(plan)
                if chess_sql:
                    sql_list.append(chess_sql)
            except Exception as e:
                logger.warning(f"CHESS generation failed: {e}")
                
            # Zero-NL2SQL
            try:
                zero_sql = self.zero_generator.generate(plan)
                if zero_sql:
                    sql_list.append(zero_sql)
            except Exception as e:
                logger.warning(f"Zero-NL2SQL generation failed: {e}")
                
            # 2. 如果只有一个SQL,直接返回
            if len(sql_list) == 1:
                return sql_list[0]
                
            # 3. 如果SQL列表为空,返回简单查询
            if not sql_list:
                return f"SELECT * FROM {plan['tables'][0]}"
                
            # 4. 使用集成方法生成最终SQL
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self._build_ensemble_prompt(query, sql_list)}],
                temperature=self.temperature
            )
            sql = rsp.choices[0].message.content.strip()
            
            # 5. 验证SQL
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.warning(f"SQL validation failed: {error}")
                # 尝试修复SQL
                sql = self._fix_sql(sql, error)
                
            # 6. 优化SQL
            sql = self._optimize_sql(sql)
            
            # 7. 最终验证
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.error(f"Final SQL validation failed: {error}")
                # 如果最终验证失败,返回最相似的SQL
                for original_sql in sql_list:
                    if self._compare_sql(sql, original_sql):
                        return original_sql
                return f"SELECT * FROM {plan['tables'][0]}"
                
            return sql
            
        except Exception as e:
            logger.error(f"SQL generation failed ({e})")
            return f"SELECT * FROM {plan['tables'][0]}"
            
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            logger.warning(f"Failed to close database connection: {e}")

# 4.6 Factory and Pipeline
class SQLGeneratorFactory:
    """SQL生成器工厂类"""
    
    @staticmethod
    def create_generator(config: SQLGeneratorConfig) -> BaseSQLGenerator:
        """创建SQL生成器"""
        try:
            if config.generation_type == SQLGenerationType.BASE_LLM:
                return BaseLLMGenerator(config)
            elif config.generation_type == SQLGenerationType.DIN_SQL:
                return DINSQLGenerator(config)
            elif config.generation_type == SQLGenerationType.MAC_SQL:
                return MACSQLGenerator(config)
            elif config.generation_type == SQLGenerationType.CHESS:
                return CHESSGenerator(config)
            elif config.generation_type == SQLGenerationType.ZERO_NL2SQL:
                return ZeroNL2SQLGenerator(config)
            elif config.generation_type == SQLGenerationType.HYBRID:
                return HybridGenerator(config)
            else:
                raise ValueError(f"Unsupported generation type: {config.generation_type}")
        except Exception as e:
            logger.error(f"Failed to create SQL generator: {e}")
            # 如果创建失败,返回基础生成器
            return BaseLLMGenerator(config)

# ════════════════════ Pipeline Orchestrator ═════════════════════════════
@dataclass
class PipelineConfig:
    query_model_path: str = "./retrieval/query_model"
    table_model_path: str = "./retrieval/table_model"
    merged_table_csv: str = "merged_table.csv"
    schema_json: str = "1000_final.json"
    top_k_tables: int = 5

    # SQL生成器配置
    sql_generator_config: SQLGeneratorConfig | None = None


class TACO_SQL_Pipeline:
    """TACO-SQL管道类"""
    
    def __init__(self, config: PipelineConfig):
        """初始化管道"""
        self.config = config
        
        # 初始化组件
        self._init_components()
        
        # 初始化日志
        self._init_logging()
        
    def _init_components(self):
        """初始化管道组件"""
        try:
            # 初始化检索模型
            self.query_model = self._load_model(self.config.query_model_path)
            self.table_model = self._load_model(self.config.table_model_path)
            
            # 加载表信息
            self.tables = self._load_tables()
            
            # 初始化SQL生成器
            self.sql_generator = SQLGeneratorFactory.create_generator(self.config.sql_generator_config)
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
            
    def _init_logging(self):
        """初始化日志"""
        try:
            # 创建日志目录
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # 配置日志
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"pipeline_{timestamp}.log"
            
            # 添加文件处理器
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
            
        except Exception as e:
            logger.warning(f"Failed to initialize logging: {e}")
            
    def _load_model(self, model_path: str) -> Any:
        """加载模型"""
        try:
            # 检查模型路径
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
                
            # 加载模型
            model = SentenceTransformer(model_path)
            model.eval()  # 设置为评估模式
            
            # 如果有GPU则使用GPU
            if torch.cuda.is_available():
                model = model.to("cuda")
                
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
            
    def _rewrite_question(self, question: str) -> str:
        """重写问题"""
        try:
            prompt = f"Please rewrite the following question to make it clearer and more standardized:\n\nOriginal question: {question}\n\nRequirements:\n1. Maintain the original intent\n2. Remove irrelevant information\n3. Standardize the expression\n4. Clarify the query intent\n5. Preserve key conditions\n6. Use standard terminology\n\nPlease return only the rewritten question."

            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional question rewriting assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            rewritten = rsp.choices[0].message.content.strip()
            logger.info(f"Question rewritten: {question} -> {rewritten}")
            return rewritten
            
        except Exception as e:
            logger.warning(f"Question rewrite failed: {e}")
            return question
            
    def _link_tables(self, question: str) -> List[Dict[str, Any]]:
        """链接相关表"""
        try:
            # 编码问题
            question_embedding = self.query_model.encode(question, convert_to_tensor=True)
            
            # 计算相似度
            similarities = []
            for table in self.tables:
                # 构建表描述
                table_desc = f"{table.get('name', '')} {table.get('description', '')}"
                for col in table.get('columns', []):
                    table_desc += f" {col.get('name', '')} {col.get('description', '')}"
                    
                # 编码表描述
                table_embedding = self.table_model.encode(table_desc, convert_to_tensor=True)
                
                # 计算余弦相似度
                similarity = torch.nn.functional.cosine_similarity(
                    question_embedding.unsqueeze(0),
                    table_embedding.unsqueeze(0)
                ).item()
                
                similarities.append((table, similarity))
                
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 返回top-k个表
            selected_tables = [table for table, _ in similarities[:self.config.top_k_tables]]
            logger.info(f"Selected tables: {[table['name'] for table in selected_tables]}")
            
            return selected_tables
            
        except Exception as e:
            logger.warning(f"Table linking failed: {e}")
            return self.tables[:self.config.top_k_tables]
            
    def _optimize_join_order(self, sql: str) -> str:
        """优化JOIN顺序"""
        try:
            # 提取JOIN语句
            join_pattern = r'(?:LEFT|RIGHT|INNER|OUTER)?\s*JOIN\s+([^\s,;]+)\s+ON\s+([^;]+)'
            joins = re.findall(join_pattern, sql, re.IGNORECASE)
            
            if not joins:
                return sql
                
            # 构建JOIN图
            join_graph = {}
            for table, condition in joins:
                # 提取JOIN条件中的表
                tables = re.findall(r'([^\s,;]+)\.', condition)
                for t in tables:
                    if t not in join_graph:
                        join_graph[t] = set()
                    join_graph[t].add(table)
                    
            # 使用贪心算法优化JOIN顺序
            def get_join_order(graph):
                visited = set()
                order = []
                
                def dfs(node):
                    visited.add(node)
                    for neighbor in graph.get(node, set()):
                        if neighbor not in visited:
                            dfs(neighbor)
                    order.append(node)
                    
                # 从每个未访问的节点开始DFS
                for node in graph:
                    if node not in visited:
                        dfs(node)
                        
                return order[::-1]  # 反转得到拓扑排序
                
            # 获取优化后的JOIN顺序
            join_order = get_join_order(join_graph)
            
            # 重新构建SQL
            if len(join_order) > 1:
                # 提取原始SELECT和WHERE部分
                select_part = re.match(r'(SELECT.*?FROM\s+[^\s,;]+)', sql, re.IGNORECASE).group(1)
                where_part = re.search(r'(WHERE.*?)(?:$|JOIN)', sql, re.IGNORECASE)
                where_part = where_part.group(1) if where_part else ""
                
                # 构建新的JOIN语句
                new_joins = []
                for i in range(len(join_order)-1):
                    table1 = join_order[i]
                    table2 = join_order[i+1]
                    # 查找对应的JOIN条件
                    for t, cond in joins:
                        if (table1 in cond and table2 in cond) or (table2 in cond and table1 in cond):
                            new_joins.append(f"JOIN {t} ON {cond}")
                            break
                            
                # 组合新的SQL
                new_sql = f"{select_part} {' '.join(new_joins)} {where_part}"
                return new_sql
                
            return sql
            
        except Exception as e:
            logger.warning(f"JOIN order optimization failed: {e}")
            return sql
            
    def _plan_query(self, question: str, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """规划查询"""
        try:
            # 构建查询计划
            plan = {
                "steps": [{"expr": question}],
                "tables": [table["name"] for table in tables],
                "schemas": {table["name"]: table["schema"] for table in tables},
                "foreign_keys": {table["name"]: table["foreign_keys"] for table in tables}
            }
            return plan
            
        except Exception as e:
            logger.warning(f"Query planning failed: {e}")
            return {
                "steps": [{"expr": question}],
                "tables": [table["name"] for table in tables[:1]],
                "schemas": {},
                "foreign_keys": {}
            }
            
    def run(self, question: str) -> str:
        """运行管道"""
        try:
            # 1. 重写问题
            rewritten_question = self._rewrite_question(question)
            logger.info(f"Rewritten question: {rewritten_question}")
            
            # 2. 链接表
            linked_tables = self._link_tables(rewritten_question)
            logger.info(f"Linked tables: {[table['name'] for table in linked_tables]}")
            
            # 3. 规划查询
            query_plan = self._plan_query(rewritten_question, linked_tables)
            logger.info(f"Query plan: {query_plan}")
            
            # 4. 生成SQL
            sql = self.sql_generator.generate(query_plan)
            logger.info(f"Generated SQL: {sql}")
            
            return sql
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            # 如果执行失败,返回简单查询
            return f"SELECT * FROM {self.tables[0]['name']}"
            
    def __del__(self):
        """清理资源"""
        try:
            # 关闭日志处理器
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        except Exception as e:
            print(f"Failed to cleanup pipeline: {e}")


class QuestionClassifier:
    """Question classifier for categorizing queries into simple, non-nested, and nested types"""
    
    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        """Initialize the classifier"""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        
    def _build_prompt(self, question: str) -> str:
        """Build classification prompt"""
        return f"Please analyze the following question and categorize it into one of these three types: 1. EASY - Simple query requiring only basic SELECT and WHERE 2. NON-NESTED - Non-nested query requiring JOINs and GROUP BY 3. NESTED - Nested query requiring subqueries or CTEs\n\nQuestion: {question}\n\nClassification Criteria:\n- EASY: Single table query with simple filtering conditions\n- NON-NESTED: Multiple table JOINs, aggregate functions, grouping, etc.\n- NESTED: Contains subqueries, CTEs, window functions, etc.\n\nPlease return only the classification result (EASY/NON-NESTED/NESTED)."

    def classify(self, question: str) -> str:
        """Classify the question"""
        try:
            # Build prompt
            prompt = self._build_prompt(question)
            
            # Call LLM for classification
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional SQL problem classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0 
            )
            
            # 获取分类结果
            category = rsp.choices[0].message.content.strip().upper()
            
            # 验证分类结果
            if category not in ["EASY", "NON-NESTED", "NESTED"]:
                logger.warning(f"Invalid category {category}, defaulting to NON-NESTED")
                return "NON-NESTED"
                
            logger.info(f"Question classified as: {category}")
            return category
            
        except Exception as e:
            logger.warning(f"Question classification failed: {e}")
            return "NON-NESTED"  # 默认返回非嵌套类型

class DatasetType(enum.Enum):
    """TACO-Benchmark dataset types."""
    US = "us"                 # US Dataset
    BEIJING = "beijing"       # Beijing Dataset
    SMARTCITY = "smartcity"   # Smart City Dataset



# ═══════════════════════════ main ═══════════════════════════════════
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="TACO-SQL Framework")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--query", type=str, help="Input query to process")
    parser.add_argument("--model", type=str, choices=[
        "base_llm", "din_sql", "mac_sql", "sft_based", 
        "codes_33b", "qwen_coder", "chess", "zero_nl2sql", "hybrid"
    ], default="base_llm", help="SQL generation model to use")
    parser.add_argument("--api_key", type=str, help="API key for LLM service")
    parser.add_argument("--base_url", type=str, help="Base URL for LLM service")
    parser.add_argument("--model_path", type=str, help="Path to SFT model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for SFT models")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for generation")
    parser.add_argument("--data_path", type=str, help="Path to data directory")
    parser.add_argument("--tables_json", type=str, help="Path to tables JSON file")
    parser.add_argument("--dataset", type=str, choices=["us", "beijing", "smartcity"], 
                       default="us", help="Dataset name (us/beijing/smartcity)")
    args = parser.parse_args()

    # 加载配置文件
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        # 创建默认配置文件
        default_config = {
            "query_model_path": "./retrieval/query_model",
            "table_model_path": "./retrieval/table_model",
            "merged_table_csv": "merged_table.csv",
            "schema_json": "1000_final.json",
            "top_k_tables": 5,
            "api_key": args.api_key or "<YOUR KEY>",
            "base_url": args.base_url or "https://api.deepseek.com",
            "model_path": args.model_path or "",
            "device": args.device,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "data_path": args.data_path or f"{args.dataset}/data",  # 根据数据集设置默认路径
            "tables_json": args.tables_json or f"{args.dataset}/data/tables.json",  # 根据数据集设置默认路径
            "dataset": args.dataset
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, allow_unicode=True)
        print(f"Created default config file at {cfg_path}")
        print("Please update the config file with your settings and run again.")
        exit()

    config = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    # 创建SQL生成器配置
    sql_generator_config = SQLGeneratorConfig(
        generation_type=SQLGenerationType(args.model),
        model_name="deepseek-chat",
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        model_path=config.get("model_path"),
        device=config.get("device", "cuda:0"),
        temperature=config.get("temperature", 0.1),
        top_p=config.get("top_p", 0.9),
        data_path=config.get("data_path"),
        tables_json_path=config.get("tables_json"),
        dataset_name=config.get("dataset", "us")  # 确保使用正确的数据集名称，默认为 us
    )

    # 创建管道配置
    pipeline_config = PipelineConfig(
        query_model_path=config.get("query_model_path", "./retrieval/query_model"),
        table_model_path=config.get("table_model_path", "./retrieval/table_model"),
        merged_table_csv=config.get("merged_table_csv", "merged_table.csv"),
        schema_json=config.get("schema_json", "1000_final.json"),
        top_k_tables=config.get("top_k_tables", 5),
        sql_generator_config=sql_generator_config
    )

    # 初始化管道
    pipeline = TACO_SQL_Pipeline(pipeline_config)

    # 处理查询
    if args.query:
        user_query = args.query
    else:
        # 示例查询
        user_query = (
            "市民反映，自己3月12号14点43分收到12123给自己发的xx区xx镇至xx镇xx道路的一个违章停车，"
            "但是自己一直在河北廊坊，车牌号：xxx，手机号：mobile_123456..."
        )

    print("\nProcessing query:", user_query)
    print(f"Using model: {args.model}")
    
    try:
        sql_statement = pipeline.run(user_query)
        print("\nGenerated SQL:\n", sql_statement)
        
        # 保存结果
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"result_{timestamp}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {user_query}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"SQL: {sql_statement}\n")
            
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"\nError processing query: {str(e)}")
        logger.error("Pipeline execution failed", exc_info=True)

