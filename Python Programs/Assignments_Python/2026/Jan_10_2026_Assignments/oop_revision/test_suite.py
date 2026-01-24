import csv


class TestSuite:
    def __init__(self, suite_name):
        self.suite_name = suite_name
        self.tests = []

    def add_test(self, test_case):
        self.tests.append(test_case)

    def run_all_tests(self):
        print(f"Running all tests in suite '{self.suite_name}' (enter Pass/Fail)")
        for test in self.tests:
            prompt = f"Result for {test.test_id} - {test.test_name} (Pass/Fail) [default skip]: "
            resp = input(prompt).strip()
            if resp == "":
                print("Skipped")
                continue
            if resp.lower() in ("pass", "fail"):
                test.execute_test(resp)
            else:
                print("Invalid input, marking as Not Executed")

    def save_results_to_csv(self, file_name):
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Test ID", "Test Name", "Module", "Status", "Automation Tool"])
            for test in self.tests:
                writer.writerow(test.to_csv_row())
        print(f"Results saved to {file_name}")

    def summary_report(self):
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t.status == "Pass")
        failed = sum(1 for t in self.tests if t.status == "Fail")
        not_executed = sum(1 for t in self.tests if t.status == "Not Executed")
        print("\nExecution Summary")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Not Executed: {not_executed}")
