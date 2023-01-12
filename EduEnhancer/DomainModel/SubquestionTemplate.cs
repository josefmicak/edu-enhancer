using Common;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DomainModel
{
    public class SubquestionTemplate
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int SubquestionTemplateId { get; set; }
        public EnumTypes.SubquestionType SubquestionType { get; set; }
        public string SubquestionText { get; set; } = default!;
        public string? ImageSource { get; set; }
        public string[] PossibleAnswerList { get; set; } = default!;
        public string[] CorrectAnswerList { get; set; } = default!;
        public double SubquestionPoints { get; set; }
        public double CorrectChoicePoints { get; set; }
        public double DefaultWrongChoicePoints { get; set; }
        public double WrongChoicePoints { get; set; }
        public int QuestionTemplateId { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public QuestionTemplate QuestionTemplate { get; set; } = default!;
    }
}