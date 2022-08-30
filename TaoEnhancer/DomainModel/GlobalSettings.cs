using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class GlobalSettings
    {
        [Key]
        public int Id { get; set; }
        public bool TestingMode { get; set; }
    }
}
