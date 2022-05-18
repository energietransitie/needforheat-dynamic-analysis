DECLARE @House AS INT;

-- Note that the operator is to compare the value is set to > but can be modified in the code
DECLARE @Parameter AS TEXT;

-- Note that the start and end datetime can be removed to filter all data
DECLARE @Start AS DATETIME;
DECLARE @End AS DATETIME;

-- Resulting in periods in which the compared condition is TRUE
SELECT
    date_start,
    date_end
FROM (
    SELECT
        ROW_NUMBER() OVER(ORDER BY timestamp) AS 'i',
        timestamp AS 'date_start',
        LEAD(timestamp) OVER(ORDER BY timestamp) AS 'date_end'
    FROM (
        SELECT
            timestamp
        FROM (
            SELECT
                m.timestamp AS 'timestamp',
                LAG(m.value) OVER(ORDER BY timestamp) AS 'prev_value',
                m.value AS 'curr_value',
                LEAD(m.value) OVER(ORDER BY timestamp) AS 'next_value'
            FROM measurement m
                JOIN property p ON p.id = m.property_id
                JOIN parameter d ON d.id = p.parameter_id
            WHERE p.house_id = @House AND
                d.name = @Parameter AND
                m.timestamp >= @Start AND
                m.timestamp <= @End
            ORDER BY m.timestamp
        ) t
        WHERE (
            curr_value > prev_value AND (
                next_value IS NULL OR NOT (next_value > curr_value)
            )
        ) OR (
            next_value > curr_value AND (
                prev_value IS NULL OR NOT (curr_value > prev_value)
            )
        )
    ) t2
) t1
WHERE 
	i % 2 = 1 AND
    date_start IS NOT NULL AND
    date_end IS NOT NULL AND
    date_end > date_start;
