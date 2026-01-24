from test_case import TestCase


class AutomatedTestCase(TestCase):
    def __init__(self, test_id, test_name, module, automation_tool, status="Not Executed"):
        super().__init__(test_id, test_name, module, status=status)
        self.automation_tool = automation_tool

    def display_test_case(self):
        print(f"Test ID: {self.test_id} | Name: {self.test_name} | Module: {self.module} | Status: {self.status} | Tool: {self.automation_tool}")

    def to_csv_row(self):
        return [self.test_id, self.test_name, self.module, self.status, self.automation_tool]
