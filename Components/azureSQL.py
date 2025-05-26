import os
import urllib
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, select, text
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


def fetch_sql_data(table_name, max_retries: int = 12, delay_seconds: int = 60):
    try:
        # Get connection string from environment variables
        #connection_string = os.environ["AZURE_SQL_CONNECTIONSTRING"]
        connection_string = "Driver={ODBC Driver 18 for SQL Server};Server=tcp:deepgreen.database.windows.net,1433;Database=DeepGreen;Uid=taltmann;Pwd=Ta750007717!0818;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_string)}", connect_args={"timeout": delay_seconds})

        # Establish connection
        for attempt in range(1, max_retries + 1):
            try:
                with engine.connect() as connection:
                    # Prepare base SELECT query
                    metadata = MetaData()
                    table = Table(table_name, metadata, autoload_with=engine)

                    # Execute query and fetch results into a pandas DataFrame
                    result = connection.execute(select(table))
                    data = pd.DataFrame(result.fetchall(), columns=result.keys())

            except OperationalError as e:
                print(f"⚠️  Attempt {attempt} failed: {e}.  Retrying in {delay_seconds}s …")
                time.sleep(delay_seconds)

    except Exception as e:
        print(f"Error fetching data: {str(e)}")

    return data

