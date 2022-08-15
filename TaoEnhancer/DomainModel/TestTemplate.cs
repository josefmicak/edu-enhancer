using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class TestTemplate
    {
        public string TestNameIdentifier { get; set; }
        [Key]
        public string TestNumberIdentifier { get; set; }
        public string Title { get; set; }
        public ICollection<QuestionTemplate> QuestionTemplateList { get; set; }
    }
}