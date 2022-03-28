namespace ViewLayer.Models
{
    public class ManageUserListModel : PageModel
    {
        private List<(string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email)> students = new List<(string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email)>();
        private List<(string roleText, List<(string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email)> students)> studentsByRoles = new List<(string roleText, List<(string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email)> students)>();
        private List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)> studentsOfTao = new List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)>();
        private List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)> studentsOfTaoNotPaired = new List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)>();
        private string[] roleTexts = new string[] { };
        private string loginEmail;
        private string role;
        private string studentNumberIdentifier;
        private string text;
        private string textClass;

        public List<(string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email)> Students { get { return students; } set { students = value; } }
        public List<(string roleText, List<(string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email)> students)> StudentsByRoles { get { return studentsByRoles; } set { studentsByRoles = value; } }
        public List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)> StudentsOfTao { get { return studentsOfTao; } set { studentsOfTao = value; } }
        public List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)> StudentsOfTaoNotPaired { get { return studentsOfTaoNotPaired; } set { studentsOfTaoNotPaired = value; } }
        public string[] RoleTexts { get { return roleTexts; } set { roleTexts = value; } }
        public string LoginEmail { get { return loginEmail; } set { loginEmail = value; } }
        public string Role { get { return role; } set { role = value; } }
        public string StudentNumberIdentifier { get { return studentNumberIdentifier; } set { studentNumberIdentifier = value; } }
        public string Text { get { return text; } set { text = value; } }
        public string TextClass { get { return textClass; } set { textClass = value; } }
    }
}