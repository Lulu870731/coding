--當日執行報表

SELECT  M.RPT_ID,RPT_CYCLE,ISSUE_DATE,RUN_TIME,AVG_TIME,LST_TIME,

CASE

     WHEN AVG_TIME > LST_TIME THEN

     CASE

         WHEN LST_TIME > 0 THEN RUN_TIME * 1.0 / LST_TIME --ELSE RUN_TIME * 1.0 / AVG_TIME END

         WHEN AVG_TIME > 0 THEN RUN_TIME * 1.0 / AVG_TIME --ELSE  RUN_TIME * 1.0 / LST_TIME END END

         ELSE NULL

     END

ELSE

     CASE

         WHEN LST_TIME > 0 THEN RUN_TIME * 1.0 / LST_TIME 

         WHEN AVG_TIME > 0 THEN RUN_TIME * 1.0 / AVG_TIME

         ELSE NULL

     END

    END MULTIPLE

FROM (

SELECT H.RPT_ID,M.RPT_CYCLE,CONVERT(NVARCHAR(10),H.ISSUE_TIME,120) ISSUE_DATE,MAX(DATEDIFF(SECOND,H.START_TIME,H.END_TIME)) RUN_TIME

  FROM  dbo.RPT_EXE_HISTORY H

INNER JOIN  dbo.RPT_DEFINE_MASTER M ON H.RPT_ID = M.RPT_ID

WHERE CONVERT(NVARCHAR(10),H.ISSUE_TIME,120) = dbo.LastWorkingDay()

  AND M.RPT_CYCLE IN ('M','D')

  AND H.USER_ID IN ('BATCH','ODS_TOOL')

  AND H.STATUS = 'SUCCESSED'

GROUP BY H.RPT_ID,M.RPT_CYCLE,H.ISSUE_TIME

) M LEFT JOIN

--月報平均值行時間

(SELECT H.RPT_ID,AVG(DATEDIFF(SECOND,H.START_TIME,H.END_TIME)) AVG_TIME

  FROM  dbo.RPT_EXE_HISTORY H

INNER JOIN  dbo.RPT_DEFINE_MASTER M ON H.RPT_ID = M.RPT_ID

WHERE CONVERT(NVARCHAR(10),H.ISSUE_TIME,120) >= CONVERT(NVARCHAR(10),DATEADD(YEAR,-1, dbo.LastWorkingDay()) )

  AND CONVERT(NVARCHAR(10),H.ISSUE_TIME,120) < dbo.LastWorkingDay()

  AND M.RPT_CYCLE = 'M'

  AND H.USER_ID IN ('BATCH','ODS_TOOL')

  AND H.STATUS = 'SUCCESSED'

GROUP BY H.RPT_ID

UNION

--日報平均值行時間

SELECT H.RPT_ID,AVG(DATEDIFF(SECOND,H.START_TIME,H.END_TIME))FROM  dbo.RPT_EXE_HISTORY H

INNER JOIN  dbo.RPT_DEFINE_MASTER M ON H.RPT_ID = M.RPT_ID

WHERE CONVERT(NVARCHAR(10),H.ISSUE_TIME,120) >= CONVERT(NVARCHAR(10),DATEADD(M,-6, dbo.LastWorkingDay()) )

  AND CONVERT(NVARCHAR(10),H.ISSUE_TIME,120) < dbo.LastWorkingDay()

  AND M.RPT_CYCLE = 'D'

  AND H.USER_ID IN ('BATCH','ODS_TOOL')

  AND H.STATUS = 'SUCCESSED'

GROUP BY H.RPT_ID) V ON M.RPT_ID = V.RPT_ID

LEFT JOIN

--上一次執行時間

(SELECT H.RPT_ID,DATEDIFF(SECOND,MAX(H.START_TIME),MAX(H.END_TIME)) LST_TIME

  FROM  dbo.RPT_EXE_HISTORY H

INNER JOIN  dbo.RPT_DEFINE_MASTER M ON H.RPT_ID = M.RPT_ID

WHERE CONVERT(NVARCHAR(10),H.ISSUE_TIME,120) < dbo.LastWorkingDay()

  AND M.RPT_CYCLE IN ('M','D')

  AND H.USER_ID IN ('BATCH','ODS_TOOL')

  AND H.STATUS = 'SUCCESSED'

GROUP BY H.RPT_ID) L ON M.RPT_ID = L.RPT_ID

INNER JOIN  db4.dbo.PERFORMANCE_MONITOR_CONFIG C ON 1=1

WHERE (RUN_TIME >= (CASE WHEN AVG_TIME = 0 THEN RUN_TIME ELSE AVG_TIME END) * AVG_TIMES OR

    RUN_TIME >= (CASE WHEN LST_TIME = 0 THEN RUN_TIME ELSE LST_TIME END) * LST_TIMES)

  AND RUN_TIME > RUN_TIMES

