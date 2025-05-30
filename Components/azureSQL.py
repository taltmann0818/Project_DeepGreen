import os
import urllib
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, select, text
from sqlalchemy.exc import OperationalError, TimeoutError, DBAPIError
import time
from sqlalchemy.exc import OperationalError

def upload_data_sql(data_to_upload, table_name, chunksize=100):
    try:
        # Get connection string from environment variables
        #connection_string = os.environ["AZURE_SQL_CONNECTIONSTRING"]
        connection_string = "Driver={ODBC Driver 18 for SQL Server};Server=tcp:deepgreen.database.windows.net,1433;Database=DeepGreen;Uid=taltmann;Pwd=Ta750007717!0818;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_string)}")

        #print("Sample data to be uploaded:")
        #print(data_to_upload.head()) # Print a small sample of the data for verification

        # Upload the dataframe to SQL
        data_to_upload.to_sql(
            name=table_name,
            con=engine,
            if_exists='append',  # 'replace' if you want to overwrite, 'append' to add to existing
            index=False,
            chunksize=chunksize
        )

        print(f"Successfully uploaded {len(data_to_upload)} records to {table_name} table")

    except Exception as e:
        print(f"Error uploading data: {str(e)}")

        # Provide additional debugging information
        if 'data_to_upload' in locals():
            print("Data sample at the time of error:")
            print(data_to_upload.head())


def fetch_sql_data(
    table_name: str,
    max_retries: int = 12,
    initial_delay: int = 10,
    backoff_factor: float = 1.5
) -> pd.DataFrame:
    """
    Fetch all rows from `table_name`, retrying on cold-start or not-available errors.
    Raises RuntimeError if final attempt still fails.
    """
    # 1. Build (or read) your connection string
    conn_str = os.environ.get(
        "AZURE_SQL_CONNECTIONSTRING",
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=tcp:deepgreen.database.windows.net,1433;"
        "Database=DeepGreen;"
        "Uid=taltmann;Pwd=Ta750007717!0818;"
        "Encrypt=yes;TrustServerCertificate=no;"
        "ConnectRetryCount=3;ConnectRetryInterval=10;"
        "Connection Timeout=30;"
    )
    quoted = urllib.parse.quote_plus(conn_str)

    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            engine = create_engine(
                f"mssql+pyodbc:///?odbc_connect={quoted}",
                connect_args={"timeout": 30},
                pool_pre_ping=True,
                pool_recycle=25 * 60
            )
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=engine)

            with engine.connect() as conn:
                result = conn.execute(select(table))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df

        except (OperationalError, TimeoutError, DBAPIError) as e:
            # If it's a DBAPIError, try to extract the tracing ID for your logs
            tracing_id = None
            if hasattr(e, "orig") and getattr(e.orig, "args", None):
                # e.orig.args might be a tuple like (sqlstate, message)
                msg = e.orig.args[1] if len(e.orig.args) > 1 else str(e.orig)
                if "session tracing ID" in msg:
                    # crude parse for the GUID in braces
                    start = msg.find("{")
                    end = msg.find("}", start)
                    tracing_id = msg[start : end + 1] if start != -1 and end != -1 else None

            print(
                "Attempt %d/%d failed (%s)%s. Retrying in %ds…",
                attempt,
                max_retries,
                e,
                f" [tracing_id={tracing_id}]" if tracing_id else "",
                delay
            )

            if attempt == max_retries:
                raise RuntimeError(
                    f"Unable to fetch '{table_name}' after {max_retries} attempts"
                ) from e

            # Dispose pool to clear any dead connections, then wait
            engine.dispose()
            time.sleep(delay)
            delay = int(delay * backoff_factor)

    # Should never reach here
    raise RuntimeError("Unexpected exit from retry loop")


