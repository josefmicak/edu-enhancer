namespace Common
{
    public class Config
    {
        public static Dictionary<string, string> Application = new Dictionary<string, string>();

        public static bool TestingMode { get; set; } = false;
    }
}
