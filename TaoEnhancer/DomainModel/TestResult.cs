using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class TestResult
    {
        [Key]
        public string TestResultIdentifier { get; set; } = default!;
        public string TestNameIdentifier { get; set; } = default!;
        public string TestNumberIdentifier { get; set; } = default!;
        public TestTemplate TestTemplate { get; set; } = default!;
        public DateTime TimeStamp { get; set; } = default!;
        public Student Student { get; set; } = default!;
        public string StudentLogin { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public ICollection<QuestionResult> QuestionResultList { get; set; } = default!;
    }
}