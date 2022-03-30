namespace Common
{
    public static class Settings
    {
        // Server URL
        public const string WindowsURL = "https://localhost:7057";
        public const string UbuntuURL = "https://vsrvfeia0h-51.vsb.cz:5000";

        // Google Sign In API
        public const string GoogleSignInClientId = "755040283949-il8fo77mu6v795aj8elvu83fomd5aqj6";
        public const string GoogleSignInClientSecret = "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";

        // File paths
        public const string WindowsPath = "C:\\xampp\\exported";
        public const string UbuntuPath = @"/home/fei/mic0378/exported";

        enum Platform
        {
            Windows,
            Ubuntu
        }

        static Platform SelectedPlatform = Platform.Windows;

        public static string GetWindowsSlash()
        {
            return "\\";
        }

        public static string GetUbuntuSlash()
        {
            return "/";
        }

        public static string GetPlatformSlash()
        {
            if (SelectedPlatform == Platform.Windows)
            {
                return GetWindowsSlash();
            }
            else
            {
                return GetUbuntuSlash();
            }
        }

        public static string GetURL()
        {
            if (SelectedPlatform == Platform.Windows)
            {
                return WindowsURL;
            }
            else
            {
                return UbuntuURL;
            }
        }

        public static string GetImageURL()
        {
            if (SelectedPlatform == Platform.Windows)
            {
                return WindowsURL + "/images/";
            }
            else
            {
                return UbuntuURL + "/images/";
            }
        }

        public static string GetSignInURL()
        {
            return GetURL() + "/Account/SignIn";
        }

        public static string GetPath()
        {
            if (SelectedPlatform == Platform.Windows)
            {
                return WindowsPath;
            }
            else
            {
                return UbuntuPath;
            }
        }

        public static string GetResultsPath()
        {
            return GetPath() + GetPlatformSlash() + "results";
        }

        public static string GetResultPath(string testNameIdentifier)
        {
            return GetResultsPath() + GetPlatformSlash() + testNameIdentifier;
        }

        public static string GetResultFilePath(string testNameIdentifier, string deliveryExecutionIdentifier)
        {
            return GetResultPath(testNameIdentifier) + GetPlatformSlash() + "delivery_execution_" + deliveryExecutionIdentifier + ".xml";
        }

        public static string GetResultResultsDataPath(string testNameIdentifier, string deliveryExecutionIdentifier)
        {
            return GetResultPath(testNameIdentifier) + GetPlatformSlash() + "delivery_execution_" + deliveryExecutionIdentifier + "Results.txt";
        }

        public static string GetStudentsPath()
        {
            return GetPath() + GetPlatformSlash() + "testtakers";
        }

        public static string GetStudentFilePath(string studentNumberIdentifier)
        {
            return GetStudentsPath() + GetPlatformSlash() + studentNumberIdentifier + ".rdf";
        }

        public static string GetStudentLoginDataPath(string loginEmail)
        {
            return GetStudentsPath() + GetPlatformSlash() + loginEmail + ".txt";
        }

        public static string GetTestsPath()
        {
            return GetPath() + GetPlatformSlash() + "tests";
        }

        public static string GetTestPath(string testNameIdentifier)
        {
            return GetTestsPath() + GetPlatformSlash() + testNameIdentifier;
        }

        public static string GetTestItemsPath(string testNameIdentifier)
        {
            return GetTestPath(testNameIdentifier) + GetPlatformSlash() + "items";
        }

        public static string GetTestItemPath(string testNameIdentifier, string itemNumberIdentifier)
        {
            return GetTestItemsPath(testNameIdentifier) + GetPlatformSlash() + itemNumberIdentifier;
        }

        public static string GetTestItemFilePath(string testNameIdentifier, string itemNumberIdentifier)
        {
            return GetTestItemPath(testNameIdentifier, itemNumberIdentifier) + GetPlatformSlash() + "qti.xml";
        }

        public static string GetTestItemPointsDataPath(string testNameIdentifier, string itemNumberIdentifier)
        {
            return GetTestItemPath(testNameIdentifier, itemNumberIdentifier) + GetPlatformSlash() + "Points.txt";
        }

        public static string GetTestTestsPath(string testNameIdentifier)
        {
            return GetTestPath(testNameIdentifier) + GetPlatformSlash() + "tests";
        }

        public static string GetTestTestPath(string testNameIdentifier, string testNumberIdentifier)
        {
            return GetTestTestsPath(testNameIdentifier) + GetPlatformSlash() + testNumberIdentifier;
        }

        public static string GetTestTestFilePath(string testNameIdentifier, string testNumberIdentifier)
        {
            return GetTestTestPath(testNameIdentifier, testNumberIdentifier) + GetPlatformSlash() + "test.xml";
        }

        public static string GetTestTestNegativePointsDataPath(string testNameIdentifier, string testNumberIdentifier)
        {
            return GetTestTestPath(testNameIdentifier, testNumberIdentifier) + GetPlatformSlash() + "NegativePoints.txt";
        }
    }
}
