using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class TestResult
    {
        [Key]
        public string TestResultIdentifier { get; set; }
        public string TestNameIdentifier { get; set; }
        public TestTemplate TestTemplate { get; set; }
        public string TimeStamp { get; set; }
        public Student Student { get; set; }
        public ICollection<QuestionResult> QuestionResultList { get; set; }
    }
}