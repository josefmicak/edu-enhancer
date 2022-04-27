namespace ViewLayer.Models
{
    public class IndexModel : PageModel
    {
        private List<(string roleText, List<(string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email)> students)> studentsByRoles = new List<(string roleText, List<(string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email)> students)>();
        private string text;
        private string textClass;
        private string signInURL;

        public List<(string roleText, List<(string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email)> students)> StudentsByRoles { get { return studentsByRoles; } set { studentsByRoles = value; } }
        public string Text { get { return text; } set { text = value; } }
        public string TextClass { get { return textClass; } set { textClass = value; } }
        public string SignInURL { get { return signInURL; } set { signInURL = value; } }
    }
}