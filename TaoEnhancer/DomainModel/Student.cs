using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class Student
    {
        [Key]
        public string StudentIdentifier { get; set; }
        public string Login { get; set; }
        public string FirstName { get; set; }
        public string LastName { get; set; }
        public string Email { get; set; }
        public string FullName()
        {
            return FirstName + " " + LastName;
        }
    }
}