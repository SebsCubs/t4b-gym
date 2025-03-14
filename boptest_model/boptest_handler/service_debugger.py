import requests
import json

class BoptestDebugger:
    def __init__(self, base_url="http://127.0.0.1:80"):
        """Initialize the debugger with the base URL of the BOPTEST server."""
        self.base_url = base_url.rstrip('/')
        self.test_id = None

    def select_testcase(self, testcase_name):
        """Select a test case and get a test ID."""
        url = f"{self.base_url}/testcases/{testcase_name}/select"
        response = requests.post(url)
        data = response.json()
        
        if data['status'] == 200:
            self.test_id = data['payload']
            print(f"Successfully selected testcase. Test ID: {self.test_id}")
        else:
            raise Exception(f"Failed to select testcase: {data['message']}")
        
        return self.test_id

    def reset_testcase(self, start_time=0, warmup_period=0):
        """Reset a running testcase to a specific start time with warmup period."""
        if not self.test_id:
            raise Exception("No test case selected. Call select_testcase first.")

        url = f"{self.base_url}/initialize/{self.test_id}"
        params = {
            'start_time': start_time,
            'warmup_period': warmup_period
        }
        
        response = requests.put(url, params=params)
        data = response.json()
        
        if data['status'] == 200:
            print(f"Successfully reset testcase to start_time={start_time} with warmup_period={warmup_period}")
        else:
            raise Exception(f"Failed to reset testcase: {data['message']}")
        
        return data

    def stop_test(self, test_id=None):
        """Stop a queued or running test.
        
        Args:
            test_id: Optional test ID. If not provided, uses the currently selected test ID.
        """
        test_id = test_id or self.test_id
        if not test_id:
            raise Exception("No test ID provided or selected.")

        url = f"{self.base_url}/stop/{test_id}"
        response = requests.put(url)
        data = response.json()
        
        if data['status'] == 200:
            print(f"Successfully stopped test with ID: {test_id}")
        else:
            raise Exception(f"Failed to stop test: {data['message']}")
        
        return data

def main():
    # Example usage
    debugger = BoptestDebugger(base_url="http://127.0.0.1:80")
    
    try:
        test_id = "aea1d873-527f-4437-a9b0-66beffdc1125"
        debugger.stop_test(test_id)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
