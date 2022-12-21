using Common;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class SubquestionTemplate
    {
        [Key]
        public string SubquestionIdentifier { get; set; } = default!;
        public EnumTypes.SubquestionType SubquestionType { get; set; }
        public string SubquestionText { get; set; } = default!;
        public string? ImageSource { get; set; }
        public string[] PossibleAnswerList { get; set; } = default!;
        public string[] CorrectAnswerList { get; set; } = default!;
        public double? SubquestionPoints { get; set; }
        public double? CorrectChoicePoints { get; set; }
        public double? DefaultWrongChoicePoints { get; set; }
        public double? WrongChoicePoints { get; set; }
        public string QuestionNumberIdentifier { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public QuestionTemplate QuestionTemplate { get; set; } = default!;
    }
}