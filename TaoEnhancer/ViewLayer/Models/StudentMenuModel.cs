namespace ViewLayer.Models
{
    public class StudentMenuModel : PageModel
    {
        private List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)> students = new List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)>();

        public List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)> Students { get { return students; } set { students = value; } }
    }
}