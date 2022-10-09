using Common;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class UserRegistration
    {
        [Key]
        public string Email { get; set; } = default!;
        public string Login { get; set; } = default!;
        public string FirstName { get; set; } = default!;
        public string LastName { get; set; } = default!;
        public EnumTypes.RegistrationState State { get; set; }
        public EnumTypes.Role Role { get; set; }
        public DateTime CreationDate { get; set; }
        public string FullName()
        {
            return FirstName + " " + LastName;
        }
        public Student? Student { get; set; }
    }
}
