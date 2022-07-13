namespace DomainModel
{
    public class SubquestionTemplate
    {
        public string SubquestionIdentifier { get; set; }
        public int SubquestionType { get; set; }
        public string SubquestionText { get; set; }
        public string ImageSource { get; set; }
        public List<string> PossibleAnswerList { get; set; }
        public List<string> CorrectAnswerList { get; set; }
    }
}