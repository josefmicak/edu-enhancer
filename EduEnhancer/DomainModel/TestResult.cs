using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DomainModel
{
    public class TestResult
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int TestResultId { get; set; }
        public int TestTemplateId { get; set; } = default!;
        public TestTemplate TestTemplate { get; set; } = default!;
        public DateTime TimeStamp { get; set; } = default!;
        public Student Student { get; set; } = default!;
        public string StudentLogin { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public ICollection<QuestionResult> QuestionResults { get; set; } = default!;
        public bool IsTestingData { get; set; } = false;
    }
}