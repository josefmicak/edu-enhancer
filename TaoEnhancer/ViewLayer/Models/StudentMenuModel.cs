using DomainModel;

namespace ViewLayer.Models
{
    public class StudentMenuModel : PageModel
    {
        private List<Student> students;

        public List<Student> Students { get { return students; } set { students = value; } }
    }
}
