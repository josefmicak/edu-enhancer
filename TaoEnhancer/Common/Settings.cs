namespace Common
{
    public static class Settings
    {
        public enum Platform
        {
            Windows,
            Ubuntu
        }

        // Testing
        public static bool Admin = false;

        // System (0 = Windows, 1 = Ubuntu)
        public static readonly Platform SelectedPlatform = Platform.Windows;

        // Server URL
        public static readonly string[] URL = new string[] {
            "https://localhost:7057",
            "https://vsrvfeia0h-51.vsb.cz:5000"
        };

        // Google Sign In API
        public static readonly string GoogleSignInClientId = "755040283949-il8fo77mu6v795aj8elvu83fomd5aqj6";
        public static readonly string GoogleSignInClientSecret = "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";

        // File paths
        public static readonly string[] Path = new string[] {
            "C:\\xampp\\exported",
            @"/home/fei/mic0378/exported"
        };
        public static readonly string[] PathSeparator = new string[] {
            "\\",
            "/"
        };

        public static string GetPathSeparator()
        {
            return PathSeparator[(int)SelectedPlatform];
        }

        public static string GetURL()
        {
            return URL[(int)SelectedPlatform];
        }

        public static string GetSignInURL()
        {
            return GetURL() + "/Account/SignIn";
        }

        public static string GetImageURL(string testNameIdentifier, string itemNumberIdentifier, string imageFilePath)
        {
            return GetURL() + "/images/" + testNameIdentifier + "/items/" + itemNumberIdentifier + "/" + imageFilePath;
        }

        public static string GetPath()
        {
            return Path[(int)SelectedPlatform];
        }

        public static string GetResultsPath()
        {
            return GetPath() + GetPathSeparator() + "results";
        }

        public static string GetResultPath(string testNameIdentifier)
        {
            return GetResultsPath() + GetPathSeparator() + testNameIdentifier;
        }

        public static string GetResultFilePath(string testNameIdentifier, string deliveryExecutionIdentifier)
        {
            return GetResultPath(testNameIdentifier) + GetPathSeparator() + "delivery_execution_" + deliveryExecutionIdentifier + ".xml";
        }

        public static string GetResultResultsDataPath(string testNameIdentifier, string deliveryExecutionIdentifier)
        {
            return GetResultPath(testNameIdentifier) + GetPathSeparator() + "delivery_execution_" + deliveryExecutionIdentifier + "Results.txt";
        }

        public static string GetStudentsPath()
        {
            return GetPath() + GetPathSeparator() + "testtakers";
        }

        public static string GetStudentFilePath(string studentNumberIdentifier)
        {
            return GetStudentsPath() + GetPathSeparator() + studentNumberIdentifier + ".rdf";
        }

        public static string GetStudentLoginDataPath(string loginEmail)
        {
            return GetStudentsPath() + GetPathSeparator() + loginEmail + ".txt";
        }

        public static string GetTestsPath()
        {
            return GetPath() + GetPathSeparator() + "tests";
        }

        public static string GetTestPath(string testNameIdentifier)
        {
            return GetTestsPath() + GetPathSeparator() + testNameIdentifier;
        }

        public static string GetTestItemsPath(string testNameIdentifier)
        {
            return GetTestPath(testNameIdentifier) + GetPathSeparator() + "items";
        }

        public static string GetTestItemPath(string testNameIdentifier, string itemNumberIdentifier)
        {
            return GetTestItemsPath(testNameIdentifier) + GetPathSeparator() + itemNumberIdentifier;
        }

        public static string GetTestItemFilePath(string testNameIdentifier, string itemNumberIdentifier)
        {
            return GetTestItemPath(testNameIdentifier, itemNumberIdentifier) + GetPathSeparator() + "qti.xml";
        }

        public static string GetTestItemPointsDataPath(string testNameIdentifier, string itemNumberIdentifier)
        {
            return GetTestItemPath(testNameIdentifier, itemNumberIdentifier) + GetPathSeparator() + "Points.txt";
        }

        public static string GetTestTestsPath(string testNameIdentifier)
        {
            return GetTestPath(testNameIdentifier) + GetPathSeparator() + "tests";
        }

        public static string GetTestTestPath(string testNameIdentifier, string testNumberIdentifier)
        {
            return GetTestTestsPath(testNameIdentifier) + GetPathSeparator() + testNumberIdentifier;
        }

        public static string GetTestTestFilePath(string testNameIdentifier, string testNumberIdentifier)
        {
            return GetTestTestPath(testNameIdentifier, testNumberIdentifier) + GetPathSeparator() + "test.xml";
        }

        public static string GetTestTestNegativePointsDataPath(string testNameIdentifier, string testNumberIdentifier)
        {
            return GetTestTestPath(testNameIdentifier, testNumberIdentifier) + GetPathSeparator() + "NegativePoints.txt";
        }
    }
}
