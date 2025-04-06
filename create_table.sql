-- Create the SNP500_1day table with the same structure as NASDAQ_1day

DECLARE @TableName NVARCHAR(128) = 'SNP600_1day';
DECLARE @SQL NVARCHAR(MAX);

SET @SQL = 'SELECT * INTO dbo.' + @TableName + ' FROM dbo.nasdaq_1day WHERE 1=0;';
EXEC sp_executesql @SQL;

-- Copy the data from NASDAQ_1day to dowjones_1day
SET @SQL = 'INSERT INTO dbo.' + @TableName + ' SELECT * FROM dbo.nasdaq_1day;';
EXEC sp_executesql @SQL;

SET @SQL = 'ALTER TABLE dbo.' + @TableName + ' ADD CONSTRAINT ' + @TableName + '_pk PRIMARY KEY (Ticker, Date)';
EXEC sp_executesql @SQL
go
