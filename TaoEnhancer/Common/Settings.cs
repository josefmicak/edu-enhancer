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
    }
}