using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

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
        //public IList<QuestionResult> QuestionResultList { get; set; }
    }
}