namespace Common
{
    public class Settings
    {
        public static string GetPath()
        {
            return "C:\\xampp\\exported";
        }

        public static string GetPathSeparator()
        {
            return "\\";
        }

        public static string GetTestsPath()
        {
            return GetPath() + GetPathSeparator() + "tests";
        }

        public static string GetResultsPath()
        {
            return GetPath() + GetPathSeparator() + "results";
        }

        public static string GetTestResultsPath(string testNameIdentifier)
        {
            return GetResultsPath() + GetPathSeparator() + testNameIdentifier;
        }

        public static string GetResultPath(string testNameIdentifier, string testResultIdentifier)
        {
            return GetTestResultsPath(testNameIdentifier) + GetPathSeparator() + "delivery_execution_" + testResultIdentifier + ".xml";
        }

        public static string GetTestPath(string testNameIdentifier)
        {
            return GetTestsPath() + GetPathSeparator() + testNameIdentifier;
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
    }
}