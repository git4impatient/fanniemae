SELECT count(*) FROM fanniemae.fannie_harp_sql_normed_p 
where label=1;  --1025385  noforclosure 1003542   fclcost 21843

drop table if exists traindata_p;
create table traindata_p stored as parquet
as select * from fannie_harp_sql_normed_p
-- 3.141592653589793238462643
where locate ( strright(loan_identifier, 1), '314592') > 1;

drop table if exists testdata_p;
create table testdata_p stored as parquet
as select * from fannie_harp_sql_normed_p
-- 3.141592653589793238462643
where locate ( strright(loan_identifier, 1), '670') > 1;


drop table if exists loanfclcost_p;

-- 1035452
create table loanfclcost_p stored as parquet as 
select loan_identifier, 
sum( 
case 
when foreclosure_costs is NULL then 0
else foreclosure_costs
end 
)/272610
fcl_costs
from loan_perf
group by loan_identifier
;

drop table if exists fannie_harp_sql_normed_p;

-- this table has a column called "label" which is a design pattern for sparkml
create table fannie_harp_sql_normed_p stored as parquet 
as select
case
when p.fcl_costs > 0 then 1
else 0
end label,
a.* , p.fcl_costs 
from loanacq_sqlnormed_p a, 
loanfclcost_p p  
where a.loan_identifier = p.loan_identifier
and -- get rid of all the nulls
a.loan2val is not NULL
and a.numborrowers is not NULL
and a.creditscore is not NULL

;

select count(*) from loanacq_sqlnormed_p
where 
debt2income  -- this is the problem, over 1million NULLS
is NULL;

select 5+NULL;
select concat ( "foo" , NULL );

select count(*) from loanacq_sqlnormed_p a , loanfclcost p  where a.loan_identifier = p.loan_identifier;
select max(fcl_costs) from loanfclcost;


select loan_identifier, 
sum( 
case 
when foreclosure_costs is NULL then 0
else foreclosure_costs
end 
)
fcl_costs
from loan_perf
group by loan_identifier
limit 100
;

-- scrub and normalize data - string indexer to be applied in spark
drop table if exists loanacq_sqlnormed_p ;
create table loanacq_sqlnormed_p stored as parquet as
select loan_identifier, 
case origination_channel
when 'R' then .1
when 'C' then .2
when 'B' then .3
end channel, 
seller_name,
original_interest_rate/7.75 intrate, 
original_upb/1402000.0 loanamt,
original_loan_to_value/ 97 loan2val,
number_of_borrowers/6 numborrowers, 
--original_debt_to_income_ratio/64 debt2income, 
borrower_credit_score_at_origination/842 creditscore, 
property_state,
origination_date
from loan_acquisition 
--limit 20
;

select property_state, count(*) from loan_acquisition
group by property_state;

select max( number_of_units ) from loan_acquisition
;

-- get numeric indicator of start date - but not a meaningful future predictor
select  to_utc_timestamp(cast(concat ( '01/',origination_date)  AS STRING), 'MM/dd/yyyy')  origyear, count(*)
from loan_acquisition
group by origyear;

--
select  date_part('year', to_timestamp(cast(concat ( '01/',origination_date)  AS STRING), 'MM/dd/yyyy') ) origyear, count(*)
from loan_acquisition
group by origyear;
--where date_part('year', to_timestamp(cast(concat ( '01/',origination_date)  AS STRING), 'MM/dd/yyyy') ) < 2005
--limit 5;

select  date_part('year', to_timestamp(cast(concat ( '',last_paid_installment_date)  AS STRING), 'MM/dd/yyyy') ) origyear, count(*)
from loan_perf
group by origyear;

-- latest perf record for each loan
--create table loanuniqueid stored as parquet as
select  concat(loan_identifier, monthly_reporting_period) mykey,
to_timestamp(cast(monthly_reporting_period AS STRING), 'MM/dd/yyyy')  
from loan_perf 

;

select  concat(loan_identifier, first_payment_date, "X") mykey,
to_timestamp(first_payment_date , 'mm/yyyy')  
from loan_acquisition 

;
select to_timestamp('01/03/2019','dd/MM/yyyy');
 select to_timestamp('Sep 25, 1984', 'MMM dd, yyyy');

-- join loan acquisition to loan perf latest perf record

select a.loan_identifier, a.borrower_credit_score_at_origination, a.property_state, p.current_loan_delinquency_status
from loan_acquisition a,
loan_perf p,
loanuniqueid u
where a.loan_identifier = p.loan_identifier
and mykey= concat(p.loan_identifier, p.monthly_reporting_period)
limit 10
;

drop table badloan;
create table badloan stored as parquet as
select distinct( loan_identifier ) 
from loan_perf where 
foreclosure_costs > 0;


select count(*)  from loan_perf; 

select count(distinct(loan_identifier)) from loan_perf where foreclosure_cost > 0
;
select count(distinct(loan_identifier)) from loan_acquisition;


create table lperf_latest_p stored as parquet as
select loan_identifier, 
current_interest_rate, 
loan_age, 
metropolitan_statistical_area_msa, 
foreclosure_date, 
principal_forgiveness_amount, 
current_loan_delinquency_status,
min( adjusted_months_to_maturity )
from loan_perf
group by loan_identifier;

create table perf_ids stored as parquet
as select distinct ( loan_identifier)
from loan_perf;

select count(*) from perf_ids p, loan_acquisition a 
where p.loan_identifier = a.loan_identifier;


