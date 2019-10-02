--create database fanniemae;
use fanniemae;

drop table if exists loan_acquisition;

create external table loan_acquisition(
loan_identifier string , 
origination_channel  string , 
seller_name  string , 
original_interest_rate decimal ( 14,10 ),
original_upb decimal ( 11,2 ),
original_loan_term int,
origination_date string, 
first_payment_date string,
original_loan_to_value decimal ( 14,10 ),
original_combined_loan_to_value decimal ( 14,10 ),
number_of_borrowers int,
original_debt_to_income_ratio decimal ( 14,10 ),
borrower_credit_score_at_origination int, 
first_time_home_buyer_indicator  string , 
loan_purpose  string , 
property_type  string , 
number_of_units  string , 
occupancy_type  string , 
property_state  string , 
zip_code string , 
primary_mortgage_insurance_percent decimal ( 14,10 ),
product_type  string , 
co_borrower_credit_score_at_origination int, 
mortgage_insurance_type int,
relocation_mortgage_indicator  string 
)
row format delimited
fields terminated by '|'
stored as textfile
location '/user/marty/fanniemae/acq'
;

select count(*) from loan_acquisition;
select * from loan_acquisition limit 1;

select 
avg ( abs( co_borrower_credit_score_at_origination -  borrower_credit_score_at_origination) )
from loan_acquisition
where  number_of_borrowers > 1 ;
