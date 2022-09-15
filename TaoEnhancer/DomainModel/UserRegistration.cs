using Common;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class UserRegistration
    {
        [Key]
        public string Email { get; set; }
        public string Login { get; set; }
        public string FirstName { get; set; }
        public string LastName { get; set; }
        public Config.RegistrationState State { get; set; }
        public Config.Role Role { get; set; }
        public DateTime CreationDate { get; set; }
        public string FullName()
        {
            return FirstName + " " + LastName;
        }
        public Student? Student { get; set; }
    }
}
