import logging
import time
from typing import Any, Dict, List, Optional, Annotated

from semantic_kernel.functions import kernel_function

from .nl2sql_types import (
    DataSourceItem,
    DataSourcesList,
    ExecuteQueryResult,
    MeasureItem,
    MeasuresList,
    QueryItem,
    QueriesRetrievalResult,
    SchemaInfo,
    TableItem,
    TableRetrievalItem,
    TablesList,
    TablesRetrievalResult,
    ValidateSQLQueryResult,
)

from connectors import (
    CosmosDBClient,
    GenAIModelClient,
    SearchClient,
    SemanticModelClient,
    SQLDBClient,
    SQLEndpointClient,
)
from connectors.types import (
    SemanticModelConfig,
    SQLDatabaseConfig,
    SQLEndpointConfig,
)

from dependencies import get_config

class NL2SQLPlugin:
    def __init__(self):
        self.cosmos = CosmosDBClient()
        self.search = SearchClient()
        cfg = get_config()
        self.container_name = cfg.get("DATASOURCES_CONTAINER", "datasources")
        self.tables_index = cfg.get("SEARCH_TABLES_INDEX_NAME", "nl2sql-tables")
        self.measures_index = cfg.get("SEARCH_MEASURES_INDEX_NAME", "nl2sql-measures")
        self.queries_index = cfg.get("SEARCH_QUERIES_INDEX_NAME", "nl2sql-queries")
        self.search_approach = cfg.get("SEARCH_APPROACH", "hybrid").lower()
        self.use_semantic = cfg.get("SEARCH_USE_SEMANTIC", "false") == "true"
        self.semantic_config = cfg.get("SEARCH_SEMANTIC_SEARCH_CONFIG", "my-semantic-config")
        self.tables_top_k = int(cfg.get("SEARCH_TABLES_INDEX_TOP_K", 10))
        self.queries_top_k = int(cfg.get("SEARCH_QUERIES_INDEX_TOP_K", 3))

    async def _perform_search(self, body: Dict[str, Any], search_index: str) -> Dict[str, Any]:
        return await self.search.search(index_name=search_index, body=body)

    @kernel_function(
        name="GetAllDatasourcesInfo",
        description=(
            "Retrieve a list of all datasources. "
            "Returns a DataSourcesList object with each datasource's name, "
            "description, and type."
        )
    )
    async def get_all_datasources_info(self) -> DataSourcesList:
        documents = await self.cosmos.list_documents(self.container_name)
        datasources = [
            DataSourceItem(
                name=doc.get("id", ""),
                description=doc.get("description", ""),
                type=doc.get("type", ""),
            )
            for doc in documents
        ]
        return DataSourcesList(datasources=datasources)

    @kernel_function(
        name="GetAllTablesInfo",
        description=(
            "Retrieve a list of tables filtering by the given datasource. "
            "Each entry will have table, description, and datasource."
        )
    )
    async def get_all_tables_info(
        self,
        datasource: Annotated[str, "Name of the target datasource"]
    ) -> TablesList:
        search_index = self.tables_index
        safe_ds = datasource.replace("'", "''")
        body = {
            "search": "*",
            "filter": f"datasource eq '{safe_ds}'",
            "select": "table, description, datasource",
            "top": 1000
        }

        logging.info(f"[tables] Querying tables for datasource '{datasource}'")
        tables_info: List[TableItem] = []
        error: Optional[str] = None

        try:
            start = time.time()
            result = await self._perform_search(body, search_index)
            logging.info(f"[tables] completed in {round(time.time()-start,2)}s")
            for doc in result.get("value", []):
                tables_info.append(
                    TableItem(
                        table=doc.get("table", ""),
                        description=doc.get("description", ""),
                        datasource=doc.get("datasource", ""),
                    )
                )
        except Exception as ex:
            error = str(ex)
            logging.error(f"[tables] error: {error}")

        if not tables_info:
            return TablesList(
                tables=[],
                error=f"No datasource named '{datasource}'. {error or ''}".strip()
            )

        return TablesList(tables=tables_info, error=error)

    @kernel_function(
        name="GetSchemaInfo",
        description=(
            "Retrieve schema information (columns and description) "
            "for a specific table in the given datasource."
        )
    )
    async def get_schema_info(
        self,
        datasource: Annotated[str, "Target datasource"],
        table_name: Annotated[str, "Target table"]
    ) -> SchemaInfo:
        search_index = self.tables_index
        safe_ds = datasource.replace("'", "''")
        safe_table = table_name.replace("'", "''")
        body = {
            "search": "*",
            "filter": f"datasource eq '{safe_ds}' and table eq '{safe_table}'",
            "select": "table, description, datasource, columns",
            "top": 1
        }

        logging.info(f"[tables] Querying schema for '{table_name}' in '{datasource}'")
        try:
            start = time.time()
            result = await self._perform_search(body, search_index)
            logging.info(f"[tables] completed in {round(time.time()-start,2)}s")
            docs = result.get("value", [])
            if not docs:
                return SchemaInfo(
                    datasource=datasource,
                    table=table_name,
                    error=f"Table '{table_name}' not found in '{datasource}'.",
                    columns=None
                )
            doc = docs[0]
            cols = {}
            for c in doc.get("columns", []):
                name = c.get("name")
                if name:
                    cols[name] = c.get("description", "")
            return SchemaInfo(
                datasource=datasource,
                table=doc.get("table", table_name),
                description=doc.get("description", ""),
                columns=cols
            )
        except Exception as ex:
            msg = str(ex)
            logging.error(f"[tables] schema error: {msg}")
            return SchemaInfo(
                datasource=datasource,
                table=table_name,
                error=msg,
                columns=None
            )

    @kernel_function(
        name="TablesRetrieval",
        description=(
            "Retrieve necessary tables based on an optimized query string. "
            "Returns a list of TableRetrievalItem and optional error."
        )
    )
    async def tables_retrieval(
        self,
        input: Annotated[str, "Optimized retrieval query"],
        datasource: Annotated[Optional[str], "Target datasource"] = None
    ) -> TablesRetrievalResult:
        idx = self.tables_index
        approach = self.search_approach
        k = self.tables_top_k
        use_semantic = self.use_semantic
        sem_cfg = self.semantic_config

        aoai = GenAIModelClient()
        logging.info("[tables] generating embeddings")
        embeddings = await aoai.get_embeddings(input)

        body: Dict[str, Any] = {"select": "table, description", "top": k}
        if datasource:
             body["filter"] = f"datasource eq '{datasource}'"
        if approach in ("term", "hybrid"):
            body["search"] = input
        if approach in ("vector", "hybrid"):
            body.setdefault("vectorQueries", []).append({
                "kind": "vector",
                "vector": embeddings,
                "fields": "contentVector",
                "k": k
            })
        if use_semantic and approach != "vector":
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = sem_cfg

        logging.info("[tables] retrieving tables")
        results: List[TableRetrievalItem] = []
        err: Optional[str] = None
        try:
            start = time.time()
            res = await self._perform_search(body, idx)
            logging.info(f"[tables] done in {round(time.time()-start,2)}s")
            for d in res.get("value", []):
                results.append(TableRetrievalItem(
                    table=d.get("table",""),
                    description=d.get("description",""),
                    datasource=d.get("datasource",None)
                ))
        except Exception as ex:
            err = str(ex)
            logging.error(f"[tables] retrieval error: {err}")

        return TablesRetrievalResult(tables=results, error=err)

    @kernel_function(
        name="MeasuresRetrieval",
        description=(
            "Retrieve measures for a given datasource, "
            "including name, description, type, source_table, data_type, source_model."
        )
    )
    async def measures_retrieval(
        self,
        datasource: Annotated[str, "Name of the target datasource"]
    ) -> MeasuresList:
        idx = self.measures_index
        safe_ds = datasource.replace("'", "''")
        body = {
            "search": "*",
            "filter": f"datasource eq '{safe_ds}'",
            "select": "name, description, datasource, type, source_table, data_type, source_model",
            "top": 1000
        }

        logging.info(f"[measures] querying for datasource '{datasource}'")
        measures: List[MeasureItem] = []
        err: Optional[str] = None
        try:
            start = time.time()
            res = await self._perform_search(body, idx)
            logging.info(f"[measures] done in {round(time.time()-start,2)}s")
            for d in res.get("value", []):
                measures.append(MeasureItem(
                    name=d.get("name",""),
                    description=d.get("description",""),
                    datasource=d.get("datasource",""),
                    type=d.get("type", "external"),
                    source_table=d.get("source_table", None),
                    data_type=d.get("data_type", None),
                    source_model=d.get("source_model", None)
                ))
        except Exception as ex:
            err = str(ex)
            logging.error(f"[measures] error: {err}")

        if not measures:
            return MeasuresList(measures=[], error=f"No measures for '{datasource}'. {err or ''}".strip())

        return MeasuresList(measures=measures, error=err)

    @kernel_function(
        name="QueriesRetrieval",
        description=(
            "Retrieve saved SQL queries (question, query, reasoning) "
            "based on the user ask and optional datasource filter."
        )
    )
    async def queries_retrieval(
        self,
        input: Annotated[str, "The user ask"],
        datasource: Annotated[str, "Target datasource name"] = None,
    ) -> QueriesRetrievalResult:
        VECTOR, TERM, HYBRID = "vector", "term", "hybrid"
        idx = self.queries_index
        approach = self.search_approach
        top_k = self.queries_top_k
        use_sem = self.use_semantic
        sem_cfg = self.semantic_config

        aoai = GenAIModelClient()
        logging.info("[queries] generating embeddings")
        embeddings = await aoai.get_embeddings(input)

        body: Dict[str, Any] = {"select": "question, query, reasoning", "top": top_k}
        if datasource:
             body["filter"] = f"datasource eq '{datasource}'"
        if approach in (TERM, HYBRID):
            body["search"] = input
        if approach in (VECTOR, HYBRID):
            body.setdefault("vectorQueries", []).append({
                "kind": "vector",
                "vector": embeddings,
                "fields": "contentVector",
                "k": top_k
            })
        if use_sem and approach != VECTOR:
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = sem_cfg

        logging.info("[queries] querying index")
        results = []
        err: Optional[str] = None
        try:
            start = time.time()
            res = await self._perform_search(body, idx)
            logging.info(f"[queries] done in {round(time.time()-start,2)}s")
            for d in res.get("value", []):
                results.append(QueryItem(
                    question=d.get("question",""),
                    query=d.get("query",""),
                    reasoning=d.get("reasoning","")
                ))
        except Exception as ex:
            err = str(ex)
            logging.error(f"[queries] error: {err}")

        return QueriesRetrievalResult(queries=results, error=err)

    @kernel_function(
        name="ValidateSQLQuery",
        description="Validate the syntax of an SQL query. Returns a ValidateSQLQueryResult indicating validity."
    )
    async def validate_sql_query(
        self,
        query: Annotated[str, "SQL Query"]
    ) -> ValidateSQLQueryResult:
        try:
            import sqlparse
            parsed = sqlparse.parse(query)
            if parsed and len(parsed) > 0:
                return ValidateSQLQueryResult(is_valid=True)
            else:
                return ValidateSQLQueryResult(is_valid=False, error="Query could not be parsed.")
        except Exception as e:
            return ValidateSQLQueryResult(is_valid=False, error=str(e))

    @kernel_function(
        name="ExecuteDAXQuery",
        description="Execute a DAX query against a semantic model datasource and return the results."
    )
    async def execute_dax_query(
        self,
        datasource: Annotated[str, "Target datasource"],
        query: Annotated[str, "DAX Query"],
        access_token: Annotated[str, "User Access Token"]
    ) -> ExecuteQueryResult:
        try:
            cosmosdb = self.cosmos
            datasource_config = await cosmosdb.get_document(self.container_name, datasource)
            if not datasource_config or datasource_config.get("type") != "semantic_model":
                return ExecuteQueryResult(error=f"{datasource} datasource configuration not found or invalid for Semantic Model.")

            semantic_model_config = SemanticModelConfig(
                id=datasource_config.get("id"),
                description=datasource_config.get("description"),
                type=datasource_config.get("type"),
                organization=datasource_config.get("organization"),
                dataset=datasource_config.get("dataset"),
                workspace=datasource_config.get("workspace"),
                tenant_id=datasource_config.get("tenant_id"),
                client_id=datasource_config.get("client_id")
            )
            semantic_client = SemanticModelClient(semantic_model_config)
            results = await semantic_client.execute_restapi_dax_query(dax_query=query, user_token=access_token)
            return ExecuteQueryResult(results=results)
        except Exception as e:
            return ExecuteQueryResult(error=str(e))

    @kernel_function(
        name="ExecuteSQLQuery",
        description="Execute a SQL query against a SQL datasource and return the results. Only SELECT statements are allowed."
    )
    async def execute_sql_query(
        self,
        datasource: Annotated[str, "Target datasource name"],
        query: Annotated[str, "SQL Query"]
    ) -> ExecuteQueryResult:
        # log entry and parameters
        logging.info(f"execute_sql_query called with datasource={datasource}, query={query}")
        try:
            cosmosdb = self.cosmos
            datasource_config = await cosmosdb.get_document(self.container_name, datasource)
            logging.debug(f"Datasource config fetched: {datasource_config}")

            if not datasource_config:
                logging.error(f"Datasource '{datasource}' configuration not found.")
                return ExecuteQueryResult(error=f"{datasource} datasource configuration not found.")

            datasource_type = datasource_config.get("type")
            # select client based on type
            if datasource_type == "sql_endpoint":
                logging.info("Initializing SQLEndpointClient")
                sql_endpoint_config = SQLEndpointConfig(
                    id=datasource_config.get("id"),
                    description=datasource_config.get("description"),
                    type=datasource_config.get("type"),
                    organization=datasource_config.get("organization"),
                    server=datasource_config.get("server"),
                    database=datasource_config.get("database"),
                    tenant_id=datasource_config.get("tenant_id"),
                    client_id=datasource_config.get("client_id")
                )
                sql_client = SQLEndpointClient(sql_endpoint_config)
            elif datasource_type == "sql_database":
                logging.info("Initializing SQLDBClient")
                sql_database_config = SQLDatabaseConfig(
                    id=datasource_config.get("id"),
                    description=datasource_config.get("description"),
                    type=datasource_config.get("type"),
                    server=datasource_config.get("server"),
                    database=datasource_config.get("database"),
                    uid=datasource_config.get("uid", None)
                )
                sql_client = SQLDBClient(sql_database_config)
            else:
                logging.error(f"Unsupported datasource type: {datasource_type}")
                return ExecuteQueryResult(error="Datasource type not supported for SQL queries.")

            logging.info("Creating SQL connection")
            connection = await sql_client.create_connection()
            cursor = connection.cursor()

            # only SELECT allowed
            if not query.strip().lower().startswith('select'):
                logging.error("Rejected non-SELECT statement")
                return ExecuteQueryResult(error="Only SELECT statements are allowed.")

            logging.info("Executing SQL query")
            cursor.execute(query)
            logging.info("Fetching query results")
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            logging.info(f"Query returned {len(rows)} rows")

            results = [dict(zip(columns, row)) for row in rows]
            return ExecuteQueryResult(results=results)

        except Exception as e:
            logging.error(f"execute_sql_query error: {e}", exc_info=True)
            return ExecuteQueryResult(error=str(e))