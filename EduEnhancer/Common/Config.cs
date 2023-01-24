namespace Common
{
    /// <summary>
    /// A class containing various custom settings that are used throughout the application
    /// </summary>
    public class Config
    {
        public static Dictionary<string, string> Application = new Dictionary<string, string>();

        public static string? GoogleClientId { get; set; }

        public static bool TestingMode { get; set; } = false;

        public static EnumTypes.Platform SelectedPlatform { get; set; } = EnumTypes.Platform.Windows;

        /// <summary>
        /// Server URL for every platform
        /// </summary>
        public static readonly string[] URL = new string[] {
            "https://localhost:7026",
            "https://vsrvfeia0h-51.vsb.cz:5000",
            //"https://localhost:5000"
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
        /// Returns Python path for every platform
        /// </summary>
        /// <returns>Python path for every platform</returns>
        public static string GetPythonPath()
        {
            return PythonPath[(int)SelectedPlatform];
        }

        /// <summary>
        /// Path to the Python executable for every platform
        /// </summary>
        public static readonly string[] PythonPath = new string[] {
            "C:\\Users\\granders\\AppData\\Local\\Programs\\Python\\Python310\\python.exe",
            "/usr/bin/python3"
        };

        /// <summary>
        /// Returns path to the solution root folder (EduEnhancer)
        /// This path can be different depending on the used OS (Windows/Ubuntu) and the used project (ViewLayer/NUnitTests)
        /// </summary>
        public static string GetSolutionRootFolderPath()
        {
            string path = Environment.CurrentDirectory;
            while (true)
            {
                if(Path.GetFileName(path) == "EduEnhancer")
                {
                    return path;
                }
                else
                {
                    path = Directory.GetParent(Environment.CurrentDirectory).FullName;
                }
            }
        }

        /// <summary>
        /// All possible path separators (Windows paths use backslashes, whereas Linux paths use forward slashes)
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
    }
}
