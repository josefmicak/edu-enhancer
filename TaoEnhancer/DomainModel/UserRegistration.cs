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
        public int State { get; set; }
        public string FullName()
        {
            return FirstName + " " + LastName;
        }
        public User? User { get; set; }
    }
}
