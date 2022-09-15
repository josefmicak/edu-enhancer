using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class TestResult
    {
        [Key]
        public string TestResultIdentifier { get; set; }
        public string TestNameIdentifier { get; set; }
        public string TestNumberIdentifier { get; set; }
        public TestTemplate TestTemplate { get; set; }
        public DateTime TimeStamp { get; set; }
        public Student Student { get; set; }
        public string StudentLogin { get; set; }
        public string OwnerLogin { get; set; }
        public ICollection<QuestionResult> QuestionResultList { get; set; }
    }
}