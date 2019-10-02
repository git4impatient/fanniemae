--create database fanniemae;
use fanniemae;

drop table if exists loan_perf;

create external table loan_perf(
loan_identifier string,
monthly_reporting_period string,
servicer_name string,
current_interest_rate decimal ( 14,10 ), 
current_actual_upb decimal ( 11,2 ), 
loan_age int, 
remaining_months_to_legal_maturity int, 
adjusted_months_to_maturity int, 
maturity_date string, 
metropolitan_statistical_area_msa string,
current_loan_delinquency_status string,
modification_flag string,
zero_balance_code string,
zero_balance_effective_date string,
last_paid_installment_date string,
foreclosure_date string,
disposition_date string,
foreclosure_costs decimal ( 27,12 ), 
property_preservation_and_repair_costs decimal ( 27,12 ),
asset_recovery_costs decimal ( 27,12 ),
miscellaneous_holding_expenses_and_credits decimal ( 27,12 ),
associated_taxes_for_holding_property        decimal  ( 27,12 ),
net_sale_proceeds decimal ( 27,12 ),
credit_enhancement_proceeds decimal ( 27,12 ),
repurchase_make_whole_proceeds decimal ( 27,12 ),
other_foreclosure_proceeds decimal ( 27,12 ),
non_interest_bearing_upb decimal ( 11,2 ),
principal_forgiveness_amount decimal ( 11,2 ),
repurchase_make_whole_proceeds_flag string,
foreclosure_principal_write_off_amount decimal ( 11,2 ),
servicing_activity_indicator string
)
row format delimited
fields terminated by '|'
stored as textfile
location '/user/marty/fanniemae/perf'
;

select count(*) from loan_perf;
select * from loan_perf limit 1;

select current_loan_delinquency_status, count(*) from loan_perf
group by current_loan_delinquency_status ;
