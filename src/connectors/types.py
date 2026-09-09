from pydantic import BaseModel
from typing import Optional

class DataSourceConfig(BaseModel):
    """
    Base configuration for a data source.

    Attributes:
        id: The unique identifier for the data source.
        description: A description of the data source.
        type: The type of the data source.
    """
    id: str
    description: str
    type: str

class SQLEndpointConfig(DataSourceConfig):
    """
    Configuration for a SQL endpoint data source.

    Attributes:
        server: The server address of the SQL endpoint.
        database: The database name.
        tenant_id: The Azure tenant ID.
        client_id: The Azure client ID.
    """
    server: str
    database: str
    tenant_id: str
    client_id: str

class SemanticModelConfig(DataSourceConfig):
    """
    Configuration for a semantic model data source.

    Attributes:
        organization: The organization name.
        workspace: The workspace identifier.
        dataset: The dataset name.
        tenant_id: The Azure tenant ID.
        client_id: The Azure client ID.
    """
    organization: str
    workspace: str
    dataset: str
    tenant_id: str
    client_id: str

class SQLDatabaseConfig(DataSourceConfig):
    """
    Configuration for a SQL database data source.

    Attributes:
        server: The server address of the SQL database.
        database: The database name.
        uid: The user ID for authentication (optional).
    """
    server: str
    database: str
    uid: Optional[str] = None    