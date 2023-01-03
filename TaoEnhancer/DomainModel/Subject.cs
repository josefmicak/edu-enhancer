using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DomainModel
{
    public class Subject
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int Id { get; set; }
        public string Abbreviation { get; set; } = default!;
        public string Name { get; set; } = default!;
        public User Guarantor { get; set; } = default!;
        public string GuarantorLogin { get; set; } = default!;
        public List<Student> StudentList { get; set; } = default!;
        public bool IsTestingData { get; set; } = false;
    }
}