namespace Common
{
    public class Config
    {
        public static Dictionary<string, string> Application = new Dictionary<string, string>();

        public static bool TestingMode { get; set; } = false;

        public static Platform SelectedPlatform { get; set; } = Platform.Windows;

        /// <summary>
        /// Platform on which the application is running (Windows/Ubuntu)
        /// </summary>
        public enum Platform
        {
            Windows,
            Ubuntu
        }

        /// <summary>
        /// Role of the user (Student/Teacher/Admin/MainAdmin)
        /// </summary>
        public enum Role
        {
            Student = 1,
            Teacher = 2,
            Admin = 3,
            MainAdmin = 4
        }

        /// <summary>
        /// State of the application (Waiting/Accepted/Rejected)
        /// </summary>
        public enum RegistrationState
        {
            Waiting,
            Accepted,
            Rejected
        }

        /// <summary>
        /// Server URL for every platform
        /// </summary>
        public static readonly string[] URL = new string[] {
            "https://localhost:7026",
            "https://vsrvfeia0h-51.vsb.cz:5000"
        };

        /// <summary>
        /// Returns server URL for every platform
        /// </summary>
        /// <returns>server URL for every platform</returns>
        public static string GetURL()
        {
            return URL[(int)SelectedPlatform];
        }

        /// <summary>
        /// Returns URL containing the page used for the Google sign in feature
        /// </summary>
        /// <returns>URL containing the page used for the Google sign in feature</returns>
        public static string GetSignInURL()
        {
            return GetURL() + "/Account/SignIn";
        }

        /// <summary>
        /// Path to the folder containing all exported files for every platform
        /// </summary>
        public static readonly string[] ExportedFilesPath = new string[] {
            "C:\\xampp\\exported",
            @"/home/fei/mic0378/exported"
        };

        /// <summary>
        /// All possible path separators (Windows paths use backslashes, whereas Ubuntu paths use forward slashes)
        /// </summary>
        public static readonly string[] PathSeparator = new string[] {
            "\\",
            "/"
        };

        /// <summary>
        /// Returns the correct path separator according to currently used platform
        /// </summary>
        /// <returns>the correct path separator according to currently used platform</returns>
        public static string GetPathSeparator()
        {
            return PathSeparator[(int)SelectedPlatform];
        }

        /// <summary>
        /// Returns the path to the folder containing all exported files according to currently used platform
        /// </summary>
        /// <returns>the path to the folder containing all exported files according to currently used platform</returns>
        public static string GetExportedFilesPath()
        {
            return ExportedFilesPath[(int)SelectedPlatform];
        }

        /// <summary>
        /// Returns the path to the folder containing all test templates
        /// </summary>
        /// <returns>the path to the folder containing all test templates</returns>
        public static string GetTestTemplatesPath()
        {
            return GetExportedFilesPath() + GetPathSeparator() + "tests";
        }

        /// <summary>
        /// Returns the path to the selected test template
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the selected test template</param>
        /// <returns>the path to the selected test template</returns>
        public static string GetTestTemplatePath(string testNameIdentifier)
        {
            return GetTestTemplatesPath() + GetPathSeparator() + testNameIdentifier;
        }

        /// <summary>
        /// Returns the path to the test.xml file of the selected test template
        /// The test.xml file contains information such as test template's title and the list of questions
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the selected test template</param>
        /// <param name="testNumberIdentifier">Number identifier of the selected test template</param>
        /// <returns>the path to the test.xml file of the selected test template</returns>
        public static string GetTestTemplateFilePath(string testNameIdentifier, string testNumberIdentifier)
        {
            return GetTestTemplatePath(testNameIdentifier) + GetPathSeparator() + "tests" + GetPathSeparator() + testNumberIdentifier + GetPathSeparator() + "test.xml";
        }

        /// <summary>
        /// Returns the path to the folder which contains question templates that belong to the select test template
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the selected test template</param>
        /// <returns>the path to the folder which contains question templates that belong to the select test template</returns>
        public static string GetQuestionTemplatesPath(string testNameIdentifier)
        {
            return GetTestTemplatePath(testNameIdentifier) + GetPathSeparator() + "items";
        }

        /// <summary>
        /// Returns the path to the qti.xml file of the selected question template
        /// The test.xml file contains information such as test template's title and the list of questions
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the selected test template</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question template</param>
        /// <returns>the path to the qti.xml file of the selected question template</returns>
        public static string GetQuestionTemplateFilePath(string testNameIdentifier, string questionNumberIdentifier)
        {
            return GetQuestionTemplatesPath(testNameIdentifier) + GetPathSeparator() + questionNumberIdentifier + GetPathSeparator() + "qti.xml";
        }

        /// <summary>
        /// Returns the path to the folder containing all test results
        /// </summary>
        /// <returns>the path to the folder containing all test results</returns>
        public static string GetResultsPath()
        {
            return GetExportedFilesPath() + GetPathSeparator() + "results";
        }

        /// <summary>
        /// Returns the path to the folder containing all test results with the selected testNameIdentifier
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the selected test</param>
        /// <returns>the path to the folder containing all test results with the selected testNameIdentifier</returns>
        public static string GetTestResultsPath(string testNameIdentifier)
        {
            return GetResultsPath() + GetPathSeparator() + testNameIdentifier;
        }

        /// <summary>
        /// Returns the path to the selected test result
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the selected test result</param>
        /// <param name="testResultIdentifier">Identifier of the selected test result</param>
        /// <returns>the path to the selected test result</returns>
        public static string GetResultPath(string testNameIdentifier, string testResultIdentifier)
        {
            return GetTestResultsPath(testNameIdentifier) + GetPathSeparator() + "delivery_execution_" + testResultIdentifier + ".xml";
        }

        /// <summary>
        /// Returns the path to the folder containing all students
        /// </summary>
        /// <returns>the path to the folder containing all students</returns>
        public static string GetStudentsPath()
        {
            return GetExportedFilesPath() + GetPathSeparator() + "testtakers";
        }
    }
}
