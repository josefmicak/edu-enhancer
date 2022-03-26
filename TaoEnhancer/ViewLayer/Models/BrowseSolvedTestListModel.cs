namespace ViewLayer.Models
{
    public class BrowseSolvedTestListModel : PageModel
    {
        private string studentIdentifier;
        private (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) student;
        private List<(string, string, string, string)> studentTestList = new List<(string, string, string, string)>();

        public string StudentIdentifier { get { return studentIdentifier; } set { studentIdentifier = value; } }
        public (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) Student { get { return student; } set { student = value; } }
        public List<(string, string, string, string)> StudentTestList { get { return studentTestList; } set { studentTestList = value; } }
    }
}