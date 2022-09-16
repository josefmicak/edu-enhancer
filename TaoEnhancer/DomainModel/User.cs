using System.ComponentModel.DataAnnotations;
using Common;

namespace DomainModel
{
    public class User
    {
        [Key]
        public string Login { get; set; } = default!;
        public string Email { get; set; } = default!;
        public string FirstName { get; set; } = default!;
        public string LastName { get; set; } = default!;
        public Config.Role Role { get; set; }
        public string FullName()
        {
            return FirstName + " " + LastName;
        }
    }
}