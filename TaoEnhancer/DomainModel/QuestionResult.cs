using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class QuestionResult
    {
        /*public string QuestionNameIdentifier { get; set; }
        public string QuestionNumberIdentifier { get; set; }*/
        public string TestResultIdentifier { get; set; }
        [Key]
        public string QuestionNumberIdentifier { get; set; }
        public TestResult TestResult { get; set; }
        public QuestionTemplate QuestionTemplate { get; set; }
        //public List<SubquestionResult> SubquestionResultList { get; set; }
    }
}