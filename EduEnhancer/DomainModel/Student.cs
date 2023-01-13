using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class Student
    {
        [Key]
        public string Login { get; set; } = default!;
        public string? Email { get; set; }
        public string FirstName { get; set; } = default!;
        public string LastName { get; set; } = default!;
        public List<Subject> Subjects { get; set; } = default!;
        public bool IsTestingData { get; set; } = false;
        public string FullName()
        {
            return FirstName + " " + LastName;
        }
    }
}