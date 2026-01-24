import csv


class TestCase:
    def __init__(self, test_id, test_name, module, status="Not Executed"):
        self.test_id = test_id
        self.test_name = test_name
        self.module = module
        self.status = status

    def execute_test(self, result):
        if isinstance(result, str) and result.strip().lower() == "pass":
            self.status = "Pass"
        else:
            self.status = "Fail"

    def display_test_case(self):
        print(f"Test ID: {self.test_id} | Name: {self.test_name} | Module: {self.module} | Status: {self.status}")

    def to_csv_row(self):
        return [self.test_id, self.test_name, self.module, self.status, "NA"]
