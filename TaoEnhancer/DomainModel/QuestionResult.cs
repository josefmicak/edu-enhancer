namespace DomainModel
{
    public class QuestionResult
    {
        public string QuestionNameIdentifier { get; set; }
        public string QuestionNumberIdentifier { get; set; }
        public List<SubquestionResult> SubquestionResultList { get; set; }
    }
}