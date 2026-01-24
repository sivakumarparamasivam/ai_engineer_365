from test_case import TestCase
from automated_test_case import AutomatedTestCase
from test_suite import TestSuite
import os


def main():
    # Create test cases
    t1 = TestCase("TC001", "Login with valid credentials", "Auth")
    t2 = TestCase("TC002", "Create new user", "User Management")
    a1 = AutomatedTestCase("AT001", "Search functionality", "Search", "Selenium")
    a2 = AutomatedTestCase("AT002", "Checkout flow", "ECommerce", "Playwright")

    # Create suite and add tests
    suite = TestSuite("Regression Cycle 1")
    for t in (t1, t2, a1, a2):
        suite.add_test(t)

    # Run tests (interactive). If run non-interactively, press Enter to skip.
    suite.run_all_tests()

    # Save results to CSV in the same folder as this script
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "test_results.csv")
    suite.save_results_to_csv(csv_path)

    # Display summary
    suite.summary_report()


if __name__ == "__main__":
    main()
