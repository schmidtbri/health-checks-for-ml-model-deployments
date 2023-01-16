"""Input and output schemas for the model."""
from pydantic import BaseModel, Field
from enum import Enum


class EmploymentLength(str, Enum):
    """Employment length in years."""
    less_than_1_year = "< 1 year"
    one_year = "1 year"
    two_years = "2 years"
    three_years = "3 years"
    four_years = "4 years"
    five_years = "5 years"
    six_years = "6 years"
    seven_years = "7 years"
    eight_years = "8 years"
    nine_years = "9 years"
    ten_years_or_more = "10+ years"


class HomeOwnership(str, Enum):
    """The home ownership status provided by the borrower during registration."""
    MORTGAGE = "MORTGAGE"
    RENT = "RENT"
    OWN = "OWN"


class LoanPurpose(str, Enum):
    """A category provided by the borrower for the loan request."""
    debt_consolidation = "debt_consolidation"
    credit_card = "credit_card"
    home_improvement = "home_improvement"
    other = "other"
    major_purchase = "major_purchase"
    small_business = "small_business"
    car = "car"
    medical = "medical"
    moving = "moving"
    vacation = "vacation"
    wedding = "wedding"
    house = "house"
    renewable_energy = "renewable_energy"
    educational = "educational"


class Term(str, Enum):
    """The number of payments on the loan."""
    thirty_six_months = " 36 months"
    sixty_months = " 60 months"


class VerificationStatus(str, Enum):
    """Indicates if income was verified."""
    source_verified = "Source Verified"
    verified = "Verified"
    not_verified = "Not Verified"


class CreditRiskModelInput(BaseModel):
    """Inputs for predicting credit risk."""
    annual_income: int = Field(ge=1896, le=273000, description="The self-reported annual income provided by the borrower during registration.")
    collections_in_last_12_months: int = Field(ge=0, le=20, description="Number of collections in 12 months excluding medical collections.")
    delinquencies_in_last_2_years: int = Field(ge=0, le=39, description="The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years.")
    debt_to_income_ratio: float = Field(ge=0.0, le=42.64, description="A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.")
    employment_length: EmploymentLength = Field(description="Employment length in years.")
    home_ownership: HomeOwnership = Field(description="The home ownership status provided by the borrower during registration.")
    number_of_delinquent_accounts: int = Field(ge=0, le=6, description="The number of accounts on which the borrower is now delinquent.")
    interest_rate: float = Field(ge=5.32, le=28.99, description="Interest rate on the loan.")
    last_payment_amount: float = Field(ge=0.0, le=36475.59, description="Last total payment amount received.")
    loan_amount: int = Field(ge=500.0, le=35000.0, description="The listed amount of the loan applied for by the borrower.")
    derogatory_public_record_count: int = Field(ge=0.0, le=86.0, description="Number of derogatory public records.")
    loan_purpose: LoanPurpose = Field(description="A category provided by the borrower for the loan request.")
    revolving_line_utilization_rate: float = Field(ge=0.0, le=892.3, description="Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.")
    term: Term = Field(description="The number of payments on the loan. Values are in months and can be either 36 or 60.")
    total_payments_to_date: float = Field(ge=0.0, le=57777.58, description="Payments received to date for portion of total amount funded by investors.")
    verification_status: VerificationStatus = Field(description="Indicates if income was verified.")


class CreditRisk(str, Enum):
    """Indicates if loan is risky."""
    safe = "safe"
    risky = "risky"


class CreditRiskModelOutput(BaseModel):
    credit_risk: CreditRisk = Field(description="Whether or not the loan is risky.")
