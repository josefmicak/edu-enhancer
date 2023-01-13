using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DomainModel
{
    public class Subject
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int SubjectId { get; set; }
        public string Abbreviation { get; set; } = default!;
        public string Name { get; set; } = default!;
        public User Guarantor { get; set; } = default!;
        public string GuarantorLogin { get; set; } = default!;
        public List<Student> Students { get; set; } = default!;
        public bool IsTestingData { get; set; } = false;
        public Subject() { }
        public Subject(string abbreviation, string name, User guarantor, string guarantorLogin, List<Student> students, bool isTestingData)
        {
            Abbreviation = abbreviation;
            Name = name;
            Guarantor = guarantor;
            GuarantorLogin = guarantorLogin;
            Students = students;
            IsTestingData = isTestingData;
        }
        public string SubjectString()
        {
            return "(" + Abbreviation + ") " + Name;
        }
    }
}