using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class SubquestionTemplate
    {
        [Key]
        public string SubquestionIdentifier { get; set; } = default!;
        public int SubquestionType { get; set; }
        public string SubquestionText { get; set; } = default!;
        public string ImageSource { get; set; } = default!;
        public string[] PossibleAnswerList { get; set; } = default!;
        public string[] CorrectAnswerList { get; set; } = default!;
        public double? SubquestionPoints { get; set; }
        public string QuestionNumberIdentifier { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public QuestionTemplate QuestionTemplate { get; set; } = default!;
    }
}