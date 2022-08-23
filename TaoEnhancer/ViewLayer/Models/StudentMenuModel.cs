using DomainModel;

namespace ViewLayer.Models
{
    public class StudentMenuModel : PageModel
    {
        private List<User> students;

        public List<User> Students { get { return students; } set { students = value; } }
    }
}
