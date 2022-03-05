namespace Common
{
    public static class Settings
    {
        public const string Path = "C:\\xampp\\exported\\";
        public const string ExternalPath = "http://testt.8u.cz/TaoEnhancer";

        public static string GetPath()
        {
            return Path;
        }

        public static string GetResultsPath()
        {
            return Path + "\\results";
        }

        public static string GetResultPath(string testNameIdentifier)
        {
            return GetResultsPath() + "\\" + testNameIdentifier;
        }

        public static string GetResultFilePath(string testNameIdentifier, string deliveryExecutionIdentifier)
        {
            return GetResultPath(testNameIdentifier) + "\\delivery_execution_" + deliveryExecutionIdentifier + ".xml";
        }

        public static string GetResultResultsDataPath(string testNameIdentifier, string deliveryExecutionIdentifier)
        {
            return GetResultPath(testNameIdentifier) + "\\delivery_execution_" + deliveryExecutionIdentifier + "Results.txt";
        }

        public static string GetStudentsPath()
        {
            return Path + "\\testtakers";
        }
        
        public static string GetStudentFilePath(string studentNumberIdentifier)
        {
            return GetStudentsPath() + "\\" + studentNumberIdentifier + ".rdf";
        }

        public static string GetTestsPath()
        {
            return Path + "\\tests";
        }

        public static string GetTestPath(string testNameIdentifier)
        {
            return GetTestsPath() + "\\" + testNameIdentifier;
        }

        public static string GetTestItemsPath(string testNameIdentifier)
        {
            return GetTestPath(testNameIdentifier) + "\\items";
        }

        public static string GetTestItemPath(string testNameIdentifier, string itemNumberIdentifier)
        {
            return GetTestItemsPath(testNameIdentifier) + "\\" + itemNumberIdentifier;
        }

        public static string GetTestItemFilePath(string testNameIdentifier, string itemNumberIdentifier)
        {
            return GetTestItemPath(testNameIdentifier, itemNumberIdentifier) + "\\qti.xml";
        }

        public static string GetTestItemPointsDataPath(string testNameIdentifier, string itemNumberIdentifier)
        {
            return GetTestItemPath(testNameIdentifier, itemNumberIdentifier) + "\\Points.txt";
        }

        public static string GetTestItemImagePath(string testNameIdentifier, string itemNumberIdentifier, string imageFilePath)
        {
            //return GetTestItemPath(testNameIdentifier, itemNumberIdentifier) + "\\" + imageFilePath;
            return ExternalPath + "/tests/" + testNameIdentifier + "/items/" + itemNumberIdentifier + "/" + imageFilePath;
        }

        public static string GetTestTestsPath(string testNameIdentifier)
        {
            return GetTestPath(testNameIdentifier) + "\\tests";
        }

        public static string GetTestTestPath(string testNameIdentifier, string testNumberIdentifier)
        {
            return GetTestTestsPath(testNameIdentifier) + "\\" + testNumberIdentifier;
        }

        public static string GetTestTestFilePath(string testNameIdentifier, string testNumberIdentifier)
        {
            return GetTestTestPath(testNameIdentifier, testNumberIdentifier) + "\\test.xml";
        }

        public static string GetTestTestNegativePointsDataPath(string testNameIdentifier, string testNumberIdentifier)
        {
            return GetTestTestPath(testNameIdentifier, testNumberIdentifier) + "\\NegativePoints.txt";
        }
    }
}
