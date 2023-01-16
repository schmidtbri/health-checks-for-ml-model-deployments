"""Prediction logic for the model."""
import os
import logging
import joblib
import pandas as pd
from ml_base import MLModel
from credit_risk_model.prediction.schemas import CreditRiskModelInput, CreditRiskModelOutput, CreditRisk
from credit_risk_model import __name__, __doc__, __version__
import zipfile

logger = logging.getLogger(__name__)


class CreditRiskModel(MLModel):
    """Prediction logic for the Credit Risk Model."""

    @property
    def display_name(self) -> str:
        """Return display name of model."""
        return "Credit Risk Model"

    @property
    def qualified_name(self) -> str:
        """Return qualified name of model."""
        return __name__

    @property
    def description(self) -> str:
        """Return description of model."""
        return __doc__

    @property
    def version(self) -> str:
        """Return version of model."""
        return __version__

    @property
    def input_schema(self):
        """Return input schema of model."""
        return CreditRiskModelInput

    @property
    def output_schema(self):
        """Return output schema of model."""
        return CreditRiskModelOutput

    def __init__(self):
        """Class constructor that loads and deserializes the model parameters."""
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        file_path = os.path.join(dir_path, "model_files", "1.zip")

        with zipfile.ZipFile(file_path) as zf:
            if "model.joblib" not in zf.namelist():
                raise ValueError("Could not find model file in zip file.")
            model_file = zf.open("model.joblib")
            self._model = joblib.load(model_file)

    def predict(self, data: CreditRiskModelInput) -> CreditRiskModelOutput:
        """Make a prediction with the model.

        Params:
            data: Data for making a prediction with the model.

        Returns:
            The result of the prediction.

        """
        if type(data) is not CreditRiskModelInput:
            raise ValueError("Input must be of type 'CreditRisk'")

        X = pd.DataFrame([[
            data.employment_length.value,
            data.home_ownership.value,
            data.loan_purpose.value,
            data.verification_status.value,
            data.term.value,
            data.annual_income,
            data.collections_in_last_12_months,
            data.delinquencies_in_last_2_years,
            data.debt_to_income_ratio,
            data.number_of_delinquent_accounts,
            data.interest_rate,
            data.last_payment_amount,
            data.loan_amount,
            data.derogatory_public_record_count,
            data.revolving_line_utilization_rate,
            data.total_payments_to_date,
        ]],
            columns=[
                "EmploymentLength",
                "HomeOwnership",
                "LoanPurpose",
                "VerificationStatus",
                "Term",
                "AnnualIncome",
                "CollectionsInLast12Months",
                "DelinquenciesInLast2Years",
                "DebtToIncomeRatio",
                "NumberOfDelinquentAccounts",
                "InterestRate",
                "LastPaymentAmount",
                "LoanAmount",
                "DerogatoryPublicRecordCount",
                "RevolvingLineUtilizationRate",
                "TotalPaymentsToDate"
            ])

        categorical_variables = ["EmploymentLength",
                                 "HomeOwnership",
                                 "LoanPurpose",
                                 "VerificationStatus",
                                 "Term"]

        for column_name in categorical_variables:
            X[column_name] = X[column_name].astype("category")

        y_hat = self._model.predict(X)[0]

        return CreditRiskModelOutput(credit_risk=CreditRisk[y_hat])
