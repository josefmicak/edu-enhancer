using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class User
    {
        [Key]
        public string Login { get; set; }
        public string? Email { get; set; }
        public string? UserIdentifier { get; set; }
        public string FirstName { get; set; }
        public string LastName { get; set; }
        public int Role { get; set; }
        public string FullName()
        {
            return FirstName + " " + LastName;
        }
    }
}